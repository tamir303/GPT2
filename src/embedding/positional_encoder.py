import errno

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    n = 10000

    def __init__(self,
                d_model,
                max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model) # (T, C)
        positions = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.pow(self.n, -torch.arange(start=0, end=d_model, step=2) / d_model)

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('pe' ,pe)

    def forward(self, x: torch.Tensor):
        try:
            return x + self.pe[:x.size(dim=1)]
        except Exception as e:
            print(f"Error {self.__class__}: {e} ")