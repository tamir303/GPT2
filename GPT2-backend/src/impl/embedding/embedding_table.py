import torch
import torch.nn as nn


class EmbeddingTable(nn.Module):
    def __init__(self,
                 d_model,
                 vocab_size):
        super().__init__()
        self.et = nn.Embedding(vocab_size, d_model)  # (V, C)
        self.unk_idx = 0

    def forward(self, x: torch.Tensor):
        try:
            invalid_mask = (x < 0) | (x >= self.et.num_embeddings)
            if invalid_mask.any():
                x = x.clone()
                x[invalid_mask] = self.unk_idx

            emb = self.et(x)  # (B, T, C)
            return emb
        except Exception as e:
            print(f"Error {self.__class__}: {e} ")
