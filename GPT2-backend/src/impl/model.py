import torch
import torch.nn as nn
from torch.nn import functional as F

from src.impl.embedding import EmbeddingTable, PositionalEncoding
from src.impl.transformer import get_transformer_decoder_block, get_transformer_encoder_block, DecoderOutputBlock

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
        self.num_layers = num_layers
        self.d_model = d_model
        self.dropout = dropout
        self.tokenizer = tokenizer

         # Block Parameters
        self.__bp_params = {
            "n_layers": num_layers,
            "n_heads": num_heads,
            "d_model": d_model,
            "d_ff": d_model // num_heads,
            "dropout": dropout,
            "block_size": max_seq_len
        }

        # Embedding for encoder and decoder
        self.__emb_enc_tokens = nn.Sequential(
            EmbeddingTable(d_model, tokenizer.get_vocab_size()),
            PositionalEncoding(d_model, max_seq_len),
        )

        # Decoder stack
        self.__decoder_stack = get_transformer_decoder_block(**self.__bp_params)

        # Decoder output block
        self.__decoder_sa = DecoderOutputBlock(d_model, tokenizer.get_vocab_size())

        self.decoder = nn.Sequential(
            self.__decoder_stack,
            self.__decoder_sa
        )

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, src_idx, targets=None):
        if src_idx is None:
            raise ValueError("src_idx cannot be None")

        src_emb = self.__emb_enc_tokens(src_idx)

        if src_emb is None:
            raise ValueError("src_emb cannot be None")

        logits = self.decoder(src_emb)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = clamp_or_replace_with_unk(targets, self.tokenizer.get_vocab_size())
            targets = targets.view(B * T)

            if logits.size(0) != targets.size(0):
                raise ValueError("logits and targets must have the same size", logits.size(), targets.size())

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, src_idx, max_new_tokens):
        self.eval()

        tgt_idx = None
        for _ in range(max_new_tokens):
            idx_cond = src_idx[:, -self.block_size:]
            logits, loss = self(idx_cond, None)
            logits = logits[:, -1, :]
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            tgt_idx = torch.cat((tgt_idx, next_token), dim=1)

        return tgt_idx


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def clamp_or_replace_with_unk(targets: torch.Tensor, vocab_size) -> torch.Tensor:
    """
    Replaces out-of-range token indices with an UNK token (ID=0).
    """
    invalid_mask = (targets < 0) | (targets >= vocab_size)
    if invalid_mask.any():
        targets = targets.clone()
        targets[invalid_mask] = 0
    return targets
