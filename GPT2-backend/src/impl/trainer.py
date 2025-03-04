import torch
import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from src.impl.config import Config
from src.impl.eval import estimate_loss
from src.impl.utils import get_batch, load_checkpoint, save_checkpoint

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 load_exiting_model: bool = False,
                 save_on_steps: bool = False,
                 device: str = "cpu",
                 log_dir: str = "runs",
                 max_grad_norm: float = 1.0,
                 patience: int = 5):

        self.params = {
            "load_checkpoint": load_exiting_model,
            "save_on_steps": save_on_steps,
            "device": device,
            "log_dir": log_dir,
            "max_grad_norm": max_grad_norm,
            "patience": patience
        }

        if load_exiting_model:
            self.current_epoch, self.best_val_loss = load_checkpoint(model, optimizer)
        else:
            self.current_epoch = 0
            self.best_val_loss = float("inf")

        self.model = model.to(device)
        self.optimizer = optimizer
        self.writer = SummaryWriter(log_dir)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: get_lr_lambda(step, warmup_steps=1000))
        self.patience_counter = 0

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)

    def train(self, data):
        self.model.train()
        for iter in tqdm.tqdm(range(self.current_epoch, Config.max_iters), desc="Training Iterations"):
            # Get source and target batches
            src, tgt = get_batch(data)
            src, tgt = src.to(self.params["device"]), tgt.to(self.params["device"])

            # Forward pass
            logits, loss = self.model(src, tgt)

            # Optimization
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.params["max_grad_norm"])
            self.optimizer.step()
            self.scheduler.step()

            # Logging
            self.writer.add_scalar('Loss/train', loss.item(), iter)
            if iter % Config.eval_interval == 0 or iter == Config.max_iters - 1:
                losses = estimate_loss(self.model, data)  # Assume estimate_loss is updated for seq2seq
                val_loss = losses['validation']
                perplexity = torch.exp(torch.tensor(val_loss)).item()

                self.writer.add_scalar('Loss/validation', val_loss, iter)
                self.writer.add_scalar('Perplexity/validation', perplexity, iter)
                tqdm.tqdm.write(
                    f"\nstep {iter}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}, val perplexity {perplexity:.4f}")

                # Checkpointing and early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    save_checkpoint(model=self.model, optimizer=self.optimizer, epoch=iter, loss=val_loss)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.params["patience"]:
                        print(f"Early stopping at iteration {iter}.")
                        break

def get_lr_lambda(step, warmup_steps=1000, max_steps=Config.max_iters):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))  # Linear warmup
    return 0.5 * (1.0 + torch.cos(torch.tensor(step - warmup_steps) * torch.pi / (max_steps - warmup_steps)))