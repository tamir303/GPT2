import math

import torch
import torch.nn.functional as F
from torch import nn


class Head(nn.Module):
    def __init__(self,
                 d_model,
                 head_size,
                 dropout,
                 block_size,
                 masked):
        super().__init__()

        self.proj_q = nn.Linear(d_model, head_size, bias=False)  # (C, *)
        self.proj_k = nn.Linear(d_model, head_size, bias=False)  # (C, *)
        self.proj_v = nn.Linear(d_model, head_size, bias=False)  # (C, *)
        self.is_masked = masked
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # (T, T)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k: torch.Tensor = self.proj_k(x)  # (B, T, C)
        q: torch.Tensor = self.proj_q(x)  # (B, T, C)

        wei = q @ k.transpose(-2, -1)  # (B, T, C) @ (B, C, T) => (B, T, T)
        wei = wei / math.sqrt(C)  # (B, T, T)
        if self.is_masked:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        v: torch.Tensor = self.proj_v(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) = (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 n_heads,
                 head_size,
                 d_model,
                 dropout,
                 block_size,
                 masked):
        super().__init__()

        self.heads = nn.ModuleList([Head(d_model, head_size, dropout, block_size, masked) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = torch.concat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
