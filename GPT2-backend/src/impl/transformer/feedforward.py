import torch
from torch import nn


class FFN(nn.Module):
    d_ff = 4

    def __init__(self,
                 d_model,
                 dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * self.d_ff),
            nn.ReLU(),
            nn.Linear(d_model * self.d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
