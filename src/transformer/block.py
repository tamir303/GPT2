import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,
                 d_model,
                 n_head):
        super().__init__()

        head_size = d_model // n_head
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffwd = None
        self.sa = nn.ModuleList([])

    def forward(self, x: torch.Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
