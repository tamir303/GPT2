import torch
import torch.nn as nn

class EmbeddingTable(nn.Module):
    def __init__(self,
                d_model,
                vocab_size):
        super().__init__()
        self.et = nn.Embedding(vocab_size, d_model) # (V, C)

    def forward(self, x: torch.Tensor):
        try:
            return self.et(x) # (B, T, C)
        except Exception as e:
            print(f"Error {self.__class__}: {e} ")