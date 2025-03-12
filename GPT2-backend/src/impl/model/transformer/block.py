from torch import nn
from src.impl.model.transformer import MultiHeadAttention, FFN

class FeedForward_Norm(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.ffn = FFN(d_model, dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.ffn(self.ln(x))


class MultiHead_Norm(nn.Module):
    def __init__(self, n_head, head_size, d_model, dropout, block_size):
        super().__init__()

        self.sa = MultiHeadAttention(n_head, head_size, d_model, dropout, block_size, False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.sa(self.ln(x))


class MaskedMultiHead_Norm(nn.Module):
    def __init__(self, n_head, head_size, d_model, dropout, block_size):
        super().__init__()

        self.sa = MultiHeadAttention(n_head, head_size, d_model, dropout, block_size, True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.sa(self.ln(x))


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, block_size):
        super().__init__()

        self.mmh_norm = MaskedMultiHead_Norm(n_head, d_model // n_head  ,d_model, dropout, block_size)
        self.ff_norm = FeedForward_Norm(d_model, dropout)

    def forward(self, x):
        x = self.mmh_norm(x)
        x = self.ff_norm(x)
        return x