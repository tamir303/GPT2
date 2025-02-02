import torch
import torch.nn as nn
from tokenizer import Tokenizer
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import tqdm

class EmbeddingTable(nn.Module):
    def __int__(self,
                d_model,
                vocab_size):
        super().__init__()
        self.et = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor):
        return self.et(x)


class PositionalEncoding(nn.Module):
    n = 10000

    def __int__(self,
                d_model,
                max_seq_len):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        positions = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.pow(self.n, -torch.arange(start=0, end=d_model, step=2) / d_model)

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('pe' ,pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class GPT2(nn.Module):
    def __int__(self,
                vocab_size,
                max_seq_len,
                d_model,
                num_heads,
                num_layers,
                batch_size,
                dropout=0.1):
        super().__init__()
