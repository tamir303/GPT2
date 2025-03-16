import torch
import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import logging

from src.etc.logger import CustomLogger
from src.etc.config import Config
from src.impl.evaluation.loss import estimate_loss
from src.impl.utils.split_batch_utils import get_batch
from src.impl.utils.load_save_utils import save_checkpoint
from src.interfaces.training import ITraining


class Trainer(ITraining):
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 device: str = Config.device,
                 log_dir: str = Config.log_dir,
                 max_grad_norm: float = Config.max_grad_norm,
                 patience: int = Config.patience,
                 warmup_steps: int = Config.warmup_steps,
                 max_steps: int = Config.max_steps,
                 identifier: str = None):
        super().__init__(model, optimizer)

        # Set up logger
        self.logger = CustomLogger(
            log_name='Training',
            log_level=logging.DEBUG,
            log_dir='app_logs',
            log_filename='trainer.log'
        ).get_logger()

        # self.logger.addHandler(handler) TODO Future log to http server

        self.params = {
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

        # if load_existing_model:
        #     try:
        #         self.current_epoch, self.best_val_loss, self.identifier = load_checkpoint(
        #             self.model, self.optimizer, save_file, storage="db"
        #         )
        #
        #         self.logger.info("Loaded checkpoint from epoch %d with best val loss %.4f",
        #                          self.current_epoch, self.best_val_loss)
        #     except FileNotFoundError:
        #         self.logger.warning("Checkpoint not found, starting from scratch.")
        #         self.current_epoch = 0
        #         self.best_val_loss = float("inf")
        #     except Exception as e:
        #         self.logger.error("Error loading checkpoint: %s", str(e))
        #         self.current_epoch = 0
        #         self.best_val_loss = float("inf")
        # else:

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.identifier = identifier

    def get_lr_lambda(self, step: int) -> float:
        if step < self.params["warmup_steps"]:
            return step / max(1, self.params["warmup_steps"])
        from math import cos, pi
        return 0.5 * (1.0 + cos((step - self.params["warmup_steps"]) *
                                  pi / (self.params["max_steps"] - self.params["warmup_steps"])))

    def train(self, x: torch.Tensor):
        self.model.train()
        self.logger.info(f"Starting training from epoch {self.current_epoch} to {Config.max_iters}")

        for iter in tqdm.tqdm(range(self.current_epoch, Config.max_iters),
                              desc="Training Iterations"):
            # Get batch and move to device
            xb, yb = get_batch(x)
            xb, yb = xb.to(self.params["device"]), yb.to(self.params["device"])

            # Forward pass
            logits, loss, acc = self.model(xb, yb)

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
                self.evaluate_and_checkpoint(x, iter, acc)

    def evaluate_and_checkpoint(self, x: torch.Tensor, iter: int, acc: float):
        # Run evaluation without computing gradients
        with torch.no_grad():
            losses = estimate_loss(self.model, x)
        val_loss = losses['validation']
        perplexity = torch.exp(torch.tensor(val_loss)).item()

        self.writer.add_scalar('Loss/validation', val_loss, iter)
        self.writer.add_scalar('Perplexity/validation', perplexity, iter)
        self.logger.info("step %d: train loss %.4f, val loss %.4f, val perplexity %.4f, accuracy %.4f ",
                         iter, losses['train'], val_loss, perplexity, acc * 100)

        # Check for improvement and save checkpoint if improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            if self.identifier:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=iter,
                    loss=val_loss,
                    accuracy=acc,
                    identifier=self.identifier,
                    storage="db"
                )
                self.logger.info("Saved checkpoint at iteration %d with val loss %.4f", iter, val_loss)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.params["patience"]:
                self.logger.info("Early stopping triggered at iteration %d", iter)
                # Optionally, you can raise an exception or break out of the training loop
                raise StopIteration

    def close(self):
        self.writer.close()
        self.logger.info("TensorBoard writer closed successfully")
