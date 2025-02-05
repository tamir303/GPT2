import torch
import torch.nn as nn
from torch.nn import functional as F

from src.embedding import EmbeddingTable, PositionalEncoding
from src.transformer import Block


class GPT2(nn.Module):
    def __init__(self,
                 tokenizer,
                 max_seq_len,
                 d_model,
                 num_heads,
                 num_layers,
                 dropout=0.1):
        super().__init__()
        self.block_size = max_seq_len

        self.token_embedding_table = EmbeddingTable(d_model, tokenizer.get_vocab_size())
        self.positional_encoder = PositionalEncoding(d_model, max_seq_len)
        self.blocks = nn.Sequential(*[Block(d_model, num_heads, dropout, max_seq_len) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_h = nn.Linear(d_model, tokenizer.get_vocab_size())  # Back to tokens

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        out = self.token_embedding_table(idx)
        out = self.positional_encoder(out)
        out = self.blocks(out)
        out = self.ln_f(out)
        logits: torch.Tensor = self.lm_h(out)  # (B, T, Vocab_size)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # Flatten to sequence of size B * T of embedded tokens
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens):
        # idx is (B, ) array of tokens
        idx = idx.unsqueeze(dim=1)
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            # focus only on last token
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1)
            # sample max prob from sample
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # apply sample in running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
