import torch
import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import logging
import os

from src.etc.logger import CustomLogger, handler
from src.etc.config import Config
from src.impl.evaluation.loss import estimate_loss
from src.impl.utils import get_batch, load_checkpoint, save_checkpoint
from src.interfaces.training import ITraining

class Trainer(ITraining):
    def __init__(self, model: nn.Module, optimizer: Optimizer, load_existing_model: bool = False,
                 save_file: str = "checkpoint.pth", device: str = "cpu", log_dir: str = "runs", max_grad_norm: float = 1.0,
                 patience: int = 5, warmup_steps: int = 1000, max_steps: int = 5000):

        super().__init__(model, optimizer)

        # Initialize logger at module level
        self.logger = CustomLogger(
            log_name='Training',
            log_level=logging.DEBUG,
            log_dir='app_logs',
            log_filename='trainer.log'
        ).get_logger()

        # Add handler to logger
        self.logger.addHandler(handler)

        self.params = {
            "load_existing_model": load_existing_model,
            "save_file": save_file,  # Now expects a filename or False
            "device": device,
            "log_dir": log_dir,
            "max_grad_norm": max_grad_norm,
            "patience": patience,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps
        }

        try:
            if torch.cuda.device_count() > 1 and "cuda" in device:
                self.logger.info(f"Using {torch.cuda.device_count()} GPUs!")
                self.model = nn.DataParallel(self.model)
            self.model = self.model.to(device)
        except RuntimeError as e:
            self.logger.error(f"Failed to initialize model on device {device}: {str(e)}")
            raise

        try:
            self.writer = SummaryWriter(log_dir)
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorBoard writer: {str(e)}")
            raise

        try:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.get_lr_lambda)
        except Exception as e:
            self.logger.error(f"Failed to initialize scheduler: {str(e)}")
            raise

        self.patience_counter = 0

        if load_existing_model:
            try:
                self.current_epoch, self.best_val_loss = load_checkpoint(model, optimizer, self.params["save_file"])
                self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch} with best val loss {self.best_val_loss:.4f}")
            except FileNotFoundError:
                self.logger.warning("Checkpoint not found, starting from scratch.")
                self.current_epoch = 0
                self.best_val_loss = float("inf")
            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {str(e)}")
                self.current_epoch = 0
                self.best_val_loss = float("inf")
        else:
            self.current_epoch = 0
            self.best_val_loss = float("inf")

    def get_lr_lambda(self, step: int) -> float:
        try:
            if step < self.params["warmup_steps"]:
                return step / max(1, self.params["warmup_steps"])
            from math import cos, pi
            return 0.5 * (1.0 + cos((step - self.params["warmup_steps"]) * pi / (self.params["max_steps"] - self.params["warmup_steps"])))
        except Exception as e:
            self.logger.error(f"Error in learning rate calculation at step {step}: {str(e)}")
            return 1.0  # Fallback to no adjustment

    def train(self, x: torch.Tensor):
        try:
            self.model.train()
            self.logger.info(f"Starting training from epoch {self.current_epoch} to {Config.max_iters}")
        except Exception as e:
            self.logger.error(f"Failed to set model to train mode: {str(e)}")
            return

        for iter in tqdm.tqdm(range(self.current_epoch, Config.max_iters), desc="Training Iterations"):
            try:
                xb, yb = get_batch(x)
                xb, yb = xb.to(self.params["device"]), yb.to(self.params["device"])
                logits, loss = self.model(xb, yb)
            except Exception as e:
                self.logger.error(f"Error in forward pass at iteration {iter}: {str(e)}")
                continue

            try:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params["max_grad_norm"])
                self.optimizer.step()
                self.scheduler.step()
            except Exception as e:
                self.logger.error(f"Error in optimization step at iteration {iter}: {str(e)}")
                continue

            try:
                self.writer.add_scalar('Loss/train', loss.item(), iter)
            except Exception as e:
                self.logger.warning(f"Failed to log training loss to TensorBoard: {str(e)}")

            if iter % Config.eval_interval == 0 or iter == Config.max_iters - 1:
                try:
                    losses = estimate_loss(self.model, x)
                    val_loss = losses['validation']
                    perplexity = torch.exp(torch.tensor(val_loss)).item()

                    self.writer.add_scalar('Loss/validation', val_loss, iter)
                    self.writer.add_scalar('Perplexity/validation', perplexity, iter)
                    self.logger.info(
                        f"step {iter}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}, val perplexity {perplexity:.4f}"
                    )

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        if self.params["save_file"]:
                            try:
                                save_checkpoint(
                                    model=self.model,
                                    optimizer=self.optimizer,
                                    epoch=iter,
                                    loss=val_loss,
                                    filename=self.params["save_file"]
                                )
                                self.logger.info(f"Saved checkpoint at iteration {iter} with val loss {val_loss:.4f}")
                            except Exception as e:
                                self.logger.error(f"Failed to save checkpoint: {str(e)}")
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.params["patience"]:
                            self.logger.info(f"Early stopping triggered at iteration {iter}")
                            break

                    # Periodic saving (assuming Config.save_interval exists)
                    if self.params["save_file"] and hasattr(Config, "save_interval") and iter % Config.save_interval == 0:
                        try:
                            periodic_filename = f"checkpoint_step_{iter}.pt"
                            save_checkpoint(
                                model=self.model,
                                optimizer=self.optimizer,
                                epoch=iter,
                                loss=val_loss,
                                filename=periodic_filename
                            )
                            self.logger.info(f"Saved periodic checkpoint: {periodic_filename}")
                        except Exception as e:
                            self.logger.error(f"Failed to save periodic checkpoint: {str(e)}")

                except Exception as e:
                    self.logger.error(f"Error during evaluation at iteration {iter}: {str(e)}")
                    continue

    def close(self):
        try:
            self.writer.close()
            self.logger.info("TensorBoard writer closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing TensorBoard writer: {str(e)}")