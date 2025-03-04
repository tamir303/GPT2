import torch
from dataclasses import dataclass

@dataclass
class Config:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 16
    block_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 100
    learning_rate: float = 1e-3
    eval_iters: int = 200
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dropout: float = 0.1
    filename: str = "checkpoint.pth"
