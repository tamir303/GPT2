import torch
from torch import nn

from src.transformer.multihead import MultiHeadAttention
from src.transformer.feedforward import FFN

class DecoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 dropout,
                 block_size,
                 first_skip):
        super().__init__()

        head_size = d_model // n_head
        self.first_skip = first_skip
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.masked_sa = MultiHeadAttention(n_head, head_size, d_model, dropout, block_size, True)
        self.sa = MultiHeadAttention(n_head, head_size, d_model, dropout, block_size, False)
        self.ffwd = FFN(d_model, dropout)

    def forward(self, x: torch.Tensor):
        if not self.first_skip:
            x = x + self.masked_sa(self.ln1(x))
            self.first_skip = False
        x = x + self.sa(self.ln2(x))
        x = x + self.ffwd(self.ln3(x))

        return x

class EncoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 dropout,
                 block_size):
        super().__init__()

        head_size = d_model // n_head
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.sa = MultiHeadAttention(n_head, head_size, d_model, dropout, block_size, False)
        self.ffwd = FFN(d_model, dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x