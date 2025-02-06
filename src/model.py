import torch
import torch.nn as nn
from torch.nn import functional as F

from src.embedding import EmbeddingTable, PositionalEncoding
from src.transformer import DecoderBlock, EncoderBlock


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

        self.emb_enc_tokens = nn.Sequential(
            EmbeddingTable(d_model, tokenizer.get_vocab_size()),
            PositionalEncoding(d_model, max_seq_len),
        )

        self.encoder = nn.Sequential(
            *[EncoderBlock(d_model, num_heads, dropout, max_seq_len) for _ in range(num_layers)]
        )

        self.decoder = nn.Sequential(
            nn.Sequential(*[DecoderBlock(d_model, num_heads, dropout, max_seq_len, first_skip = i == 0) for i in range(num_layers)]),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, tokenizer.get_vocab_size()) # Back to tokens
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, use_encoder : bool = True):
        out = idx
        if use_encoder:
            out = self.encoder(out)
        logits = self.decoder(out) # (B, T, Vocab_size)

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
        # use attention to encode possible next id
        first_idx_flag = True
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond, first_idx_flag)
            first_idx_flag = False
            # focus only on last token
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1)
            # sample max prob from sample
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # apply sample in running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
