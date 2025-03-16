from typing import Callable

import torch
import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import logging

from src.etc.logger import CustomLogger
from src.etc.config import Config
from src.impl.evaluation.loss import estimate_loss
from src.impl.utils.split_batch_utils import get_batch
from src.interfaces.training import ITraining


class Trainer(ITraining):
    def close(self) -> None:
        pass

    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 device: str = Config.device,
                 log_dir: str = Config.log_dir,
                 max_grad_norm: float = Config.max_grad_norm,
                 patience: int = Config.patience,
                 warmup_steps: int = Config.warmup_steps,
                 max_steps: int = Config.max_steps):
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

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.get_lr_lambda)
        self.patience_counter = 0

    def get_lr_lambda(self, step: int) -> float:
        if step < self.params["warmup_steps"]:
            return step / max(1, self.params["warmup_steps"])
        from math import cos, pi
        return 0.5 * (1.0 + cos((step - self.params["warmup_steps"]) *
                                  pi / (self.params["max_steps"] - self.params["warmup_steps"])))

    def train(self, x: torch.Tensor, *args, **kwargs):
        self.model.train()

        current_epoch: int = kwargs.get("current_epoch", 0)
        best_val_loss: float = kwargs.get("current_loss", float('inf'))
        save_callable: Callable | None = kwargs.get("save_callable", None)

        self.logger.info(f"Starting training from epoch {current_epoch} to {Config.max_iters}")

        for iter in tqdm.tqdm(range(current_epoch, Config.max_iters),
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

            # Evaluate model and save checkpoints periodically
            if iter % Config.eval_interval == 0 or iter == Config.max_iters - 1:
                best_val_loss = self.evaluate_and_checkpoint(x, iter, acc, best_val_loss, save_callable)

    def evaluate_and_checkpoint(self, x: torch.Tensor, iter: int, acc: float, best_val_loss: float, save_callable: Callable) -> float:
        # Run evaluation without computing gradients
        with torch.no_grad():
            losses = estimate_loss(self.model, x)

        val_loss = losses['validation']
        perplexity = torch.exp(torch.tensor(val_loss)).item()

        self.logger.info("step %d: train loss %.4f, val loss %.4f, val perplexity %.4f, accuracy %.4f ",
                         iter, losses['train'], val_loss, perplexity, acc * 100)

        # Check for improvement and save checkpoint if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            self.patience_counter = 0
            if save_callable is not None:
                save_callable(iter, losses, acc * 100)
            self.logger.info("Saved checkpoint at iteration %d with val loss %.4f", iter, val_loss)
            return best_val_loss
        else:
            if val_loss != best_val_loss: self.patience_counter += 1
            if self.patience_counter >= self.params["patience"]:
                self.logger.info("Early stopping triggered at iteration %d", iter)
                raise StopIteration
            else:
                return best_val_loss
