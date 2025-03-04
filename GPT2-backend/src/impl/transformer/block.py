from types import NoneType

from torch import nn

from src.impl.transformer.multihead import MultiHeadAttention
from src.impl.transformer.feedforward import FFN

class EncoderBlock(nn.Module):
    def __init__(self, n_head, head_size, d_model, dropout, block_size):
        super().__init__()

        self.mha = MultiHead_Norm(n_head, head_size, d_model, dropout, block_size)
        self.ffn = FeedForward_Norm(d_model, dropout)

    def forward(self, x):
        x = self.mha(x)
        return self.ffn(x)


class DecoderBlock(nn.Module):
    def __init__(self, n_head, head_size, d_model, dropout, block_size):
        super().__init__()

        self.mma = MaskedMultiHead_Norm(n_head, head_size, d_model, dropout, block_size)
        self.mha = MultiHead_Norm(n_head, head_size, d_model, dropout, block_size)
        self.ffn = FeedForward_Norm(d_model, dropout)

    def forward(self, x):
        x = self.mma(x)
        x = self.mha(x)
        return self.ffn(x)

class DecoderOutputBlock(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.lin = nn.Linear(d_model, vocab_size)  # Back to tokens

    def forward(self, x):
        return self.lin(self.ln(x))


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