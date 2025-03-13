import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.etc.logger import CustomLogger
from src.impl.model import EmbeddingTable, PositionalEncoding, Block
from src.interfaces.model import IModel


class GPT2(IModel):
    def __init__(self,
                 vocab_size: int,
                 block_size: int,
                 d_model: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float = 0.1):
        super().__init__()

        # Set up the logger
        self.logger = CustomLogger(
            log_name='GPT2',
            log_level=logging.DEBUG,
            log_dir='app_logs',
            log_filename='gpt2.log'
        ).get_logger()

        # Save hyperparameters
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Define multi-head attention parameters for blocks
        self.mh_params = {
            "n_head": self.num_heads,
            "d_model": self.d_model,
            "dropout": self.dropout,
            "block_size": self.block_size
        }

        # Initialize components
        self.emb_enc_tokens = EmbeddingTable(self.vocab_size, self.d_model)
        self.emb_pos = PositionalEncoding(self.d_model, self.block_size)
        self.blocks = nn.Sequential(*[Block(**self.mh_params) for _ in range(self.num_layers)])
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

        self.logger.info(
            "Initialized GPT2 model with vocab_size=%d, block_size=%d, d_model=%d, num_heads=%d, num_layers=%d",
            vocab_size, block_size, d_model, num_heads, num_layers
        )

        # Initialize model weights
        self.apply(self._init_weights)
        self.logger.debug("Model weights initialized successfully")

        # Precompute positional encoding once (on CPU) and register as a buffer
        with torch.no_grad():
            pos_enc = self.emb_pos(torch.arange(self.block_size, device=torch.device("cpu")))
        self.register_buffer("pos_encoding", pos_enc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        if T > self.block_size:
            self.logger.warning(
                "Input sequence length %d exceeds block_size %d, truncating", T, self.block_size
            )
            idx = idx[:, -self.block_size:]
            T = self.block_size

        # Token embedding
        tok_emb = self.emb_enc_tokens(idx)  # (B, T, C)

        # Use precomputed positional encoding (move to the current device)
        pos_emb = self.pos_encoding[:T].unsqueeze(0).expand(B, -1, -1).to(idx.device)  # (B, T, C)

        # Combine embeddings and apply transformer blocks
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)     # (B, T, C)
        x = self.ln_f(x)       # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # If targets are provided, compute loss
        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = targets.view(B * T)
        loss = F.cross_entropy(logits_flat, targets_flat)
        self.logger.debug("Forward pass: batch_size=%d, seq_len=%d, loss=%.4f", B, T, loss.item())
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        self.logger.info("Generating %d new tokens", max_new_tokens)
        for _ in range(max_new_tokens):
            # Use only the last block_size tokens as context
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Get the logits of the last token and sample the next token
            logits_last = logits[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.logger.info("Generation completed, output sequence length: %d", idx.shape[1])
        return idx
