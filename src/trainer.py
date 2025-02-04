import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

from sklearn.metrics import accuracy_score

from src import get_batch
from src import Config
from src import estimate_loss
from src import load_checkpoint, save_checkpoint

import tqdm

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 load_exiting_model: bool = False,
                 save_on_steps: bool = False,
                 device: str = "cpu",
                 log_dir:str = "runs",
                 max_grad_norm: float = 1.0,
                 patience: int = 5):

        self.model = model
        self.optimizer = optimizer
        self.writer = SummaryWriter(log_dir)

        # Learning rate scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

        # Early stopping parameters
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Other params
        self.params = {
            "load_checkpoint": load_exiting_model,
            "save_on_steps": save_on_steps,
            "device": device,
            "log_dir": log_dir,
            "max_grad_norm": max_grad_norm,
            "patience": patience
        }

        # Check if we have multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)  # Use multiple GPUs

    def train(self, x: torch.Tensor):
        for iter in tqdm.tqdm(range(Config.max_iters), desc="Training Iterations", dynamic_ncols=True):
            logits: torch.Tensor
            loss: torch.Tensor

            # sample random batch of data
            xb, yb = get_batch(x)
            xb, yb = xb.to(self.params["device"]), yb.to(self.params["device"])

            # forward pass
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # gradient clipping
            if self.params["max_grad_norm"]:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params["max_grad_norm"])

            self.optimizer.step()
            self.writer.add_scalar('Loss/train', loss.item(), iter)

            if iter % Config.eval_interval == 0 or iter == Config.max_iters - 1:
                losses = estimate_loss(self.model, x)
                val_loss = losses['validation']

                y_true = yb.view(-1).cpu().numpy()
                y_pred = logits.argmax(dim=-1).view(-1).cpu().numpy()
                acc = accuracy_score(y_true, y_pred)

                # Log validation loss and accuracy
                self.writer.add_scalar('Loss/validation', val_loss, iter)
                self.writer.add_scalar('Accuracy/validation', acc, iter)

                tqdm.tqdm.write(f"\nstep {iter}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}, val acc {acc:.4f}")

                # Early stopping logic
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0

                    # Save best checkpoint
                    save_checkpoint(iter, self.model, self.optimizer, val_loss, loss)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.params["patience"]:
                        print(f"\nEarly stopping at iteration {iter}.")
                        return  # Stop training early


            self.scheduler.step()
