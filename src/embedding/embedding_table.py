import torch
import torch.nn as nn

class EmbeddingTable(nn.Module):
    def __init__(self,
                d_model,
                vocab_size):
        super().__init__()
        self.et = nn.Embedding(d_model, vocab_size) # (V, C)

    def forward(self, x: torch.Tensor):
        return self.et(x) # (B, T, C)