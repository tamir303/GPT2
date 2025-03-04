import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    n = 10000

    def __init__(self,
                 d_model,
                 max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)  # (T, C)
        positions = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.pow(self.n, -torch.arange(start=0, end=d_model, step=2) / d_model)

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        try:
            block_len = x.size(1)  # Get the actual block length of x

            if block_len < self.pe.size(0):  # Check if x is shorter than max_seq_len
                pad_size = self.pe.size(0) - block_len  # How much to pad
                x = F.pad(x, (0, 0, 0, pad_size), value=0)  # Pad along sequence length dimension

            pe = self.pe[:x.size(1)].unsqueeze(0)
            return x + pe
        except Exception as e:
            print(f"Error {self.__class__}: {e} ")
