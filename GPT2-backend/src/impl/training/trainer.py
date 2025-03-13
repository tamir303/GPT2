import torch
import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import logging

from src.etc.logger import CustomLogger, handler
from src.etc.config import Config
from src.impl.evaluation.loss import estimate_loss
from src.impl.utils import get_batch, load_checkpoint, save_checkpoint
from src.interfaces.training import ITraining


class Trainer(ITraining):
    def __init__(self, model: nn.Module, optimizer: Optimizer, load_existing_model: bool = False,
                 save_file: str = "checkpoint.pth", device: str = "cpu", log_dir: str = "runs",
                 max_grad_norm: float = 1.0, patience: int = 5, warmup_steps: int = 1000, max_steps: int = 5000):
        super().__init__(model, optimizer)

        # Set up logger
        self.logger = CustomLogger(
            log_name='Training',
            log_level=logging.DEBUG,
            log_dir='app_logs',
            log_filename='trainer.log'
        ).get_logger()
        self.logger.addHandler(handler)

        self.params = {
            "load_existing_model": load_existing_model,
            "save_file": save_file,
            "device": device,
            "log_dir": log_dir,
            "max_grad_norm": max_grad_norm,
            "patience": patience,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps
        }

        # If multiple GPUs are available, wrap the model with DataParallel
        if torch.cuda.device_count() > 1 and "cuda" in device:
            self.logger.info("Using %d GPUs!", torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device)

        self.writer = SummaryWriter(log_dir)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.get_lr_lambda)

        self.patience_counter = 0

        if load_existing_model:
            try:
                self.current_epoch, self.best_val_loss = load_checkpoint(
                    self.model, self.optimizer, save_file
                )
                self.logger.info("Loaded checkpoint from epoch %d with best val loss %.4f",
                                 self.current_epoch, self.best_val_loss)
            except FileNotFoundError:
                self.logger.warning("Checkpoint not found, starting from scratch.")
                self.current_epoch = 0
                self.best_val_loss = float("inf")
            except Exception as e:
                self.logger.error("Error loading checkpoint: %s", str(e))
                self.current_epoch = 0
                self.best_val_loss = float("inf")
        else:
            self.current_epoch = 0
            self.best_val_loss = float("inf")

    def get_lr_lambda(self, step: int) -> float:
        if step < self.params["warmup_steps"]:
            return step / max(1, self.params["warmup_steps"])
        from math import cos, pi
        return 0.5 * (1.0 + cos((step - self.params["warmup_steps"]) *
                                  pi / (self.params["max_steps"] - self.params["warmup_steps"])))

    def train(self, x: torch.Tensor):
        self.model.train()
        self.logger.info("Starting training from epoch %d to %d",
                         self.current_epoch, Config.max_iters)

        for iter in tqdm.tqdm(range(self.current_epoch, Config.max_iters),
                              desc="Training Iterations"):
            # Get batch and move to device
            xb, yb = get_batch(x)
            xb, yb = xb.to(self.params["device"]), yb.to(self.params["device"])

            # Forward pass
            logits, loss = self.model(xb, yb)

            # Optimization step
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.params["max_grad_norm"])
            self.optimizer.step()
            self.scheduler.step()

            # Log training loss
            self.writer.add_scalar('Loss/train', loss.item(), iter)

            # Evaluate model and save checkpoints periodically
            if iter % Config.eval_interval == 0 or iter == Config.max_iters - 1:
                self.evaluate_and_checkpoint(x, iter)

    def evaluate_and_checkpoint(self, x: torch.Tensor, iter: int):
        # Run evaluation without computing gradients
        with torch.no_grad():
            losses = estimate_loss(self.model, x)
        val_loss = losses['validation']
        perplexity = torch.exp(torch.tensor(val_loss)).item()

        self.writer.add_scalar('Loss/validation', val_loss, iter)
        self.writer.add_scalar('Perplexity/validation', perplexity, iter)
        self.logger.info("step %d: train loss %.4f, val loss %.4f, val perplexity %.4f",
                         iter, losses['train'], val_loss, perplexity)

        # Check for improvement and save checkpoint if improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            if self.params["save_file"]:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=iter,
                    loss=val_loss,
                    filename=self.params["save_file"]
                )
                self.logger.info("Saved checkpoint at iteration %d with val loss %.4f", iter, val_loss)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.params["patience"]:
                self.logger.info("Early stopping triggered at iteration %d", iter)
                # Optionally, you can raise an exception or break out of the training loop
                raise StopIteration

        # Periodic checkpoint saving (if configured)
        if self.params["save_file"] and hasattr(Config, "save_interval") and iter % Config.save_interval == 0:
            periodic_filename = f"checkpoint_step_{iter}.pt"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=iter,
                loss=val_loss,
                filename=periodic_filename
            )
            self.logger.info("Saved periodic checkpoint: %s", periodic_filename)

    def close(self):
        self.writer.close()
        self.logger.info("TensorBoard writer closed successfully")
