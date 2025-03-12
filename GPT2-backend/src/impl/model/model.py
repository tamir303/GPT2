import torch
import torch.nn as nn
from torch.nn import functional as F
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

        # Initialize logger at module level
        self.logger = CustomLogger(
            log_name='GPT2',
            log_level=logging.DEBUG,
            log_dir='app_logs',
            log_filename='gpt2.log'
        ).get_logger()

        self.num_heads = num_heads
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.mh_params = {
            "n_head": self.num_heads,
            "d_model": self.d_model,
            "dropout": self.dropout,
            "block_size": self.block_size
        }

        self.ffd_params = {
            "d_model": self.d_model,
            "dropout": self.dropout
        }

        try:
            self.emb_enc_tokens = EmbeddingTable(self.vocab_size, self.d_model)
            self.emb_pos = PositionalEncoding(self.d_model, self.block_size)
            self.blocks = nn.Sequential(*[Block(**self.mh_params) for _ in range(self.num_layers)])
            self.ln_f = nn.LayerNorm(self.d_model)
            self.lm_head = nn.Linear(self.d_model, self.vocab_size)
            self.logger.info(f"Initialized GPT2 model with vocab_size={vocab_size}, block_size={block_size}, "
                       f"d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}")
        except Exception as e:
            self.logger.error(f"Failed to initialize GPT2 components: {str(e)}")
            raise

        try:
            self.apply(self._init_weights)
            self.logger.debug("Model weights initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize weights: {str(e)}")
            raise

    def _init_weights(self, module):
        try:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        except Exception as e:
            self.logger.error(f"Error in weight initialization: {str(e)}")
            raise

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        try:
            B, T = idx.shape
            if T > self.block_size:
                self.logger.warning(f"Input sequence length {T} exceeds block_size {self.block_size}, truncating")
                idx = idx[:, -self.block_size:]

            tok_emb = self.emb_enc_tokens(idx)  # (B, T, C)
            pos_emb = self.emb_pos(torch.arange(T, device=idx.device))  # (T, C)
            x = tok_emb + pos_emb  # (B, T, C)
            x = self.blocks(x)  # (B, T, C)
            x = self.ln_f(x)  # (B, T, C)
            logits = self.lm_head(x)  # (B, T, vocab_size)

            if targets is None:
                return logits, None

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            self.logger.debug(f"Forward pass completed: batch_size={B}, seq_len={T}, loss={loss.item() if loss is not None else 'None'}")
            return logits, loss

        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            return None, None

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        try:
            self.logger.info(f"Generating {max_new_tokens} new tokens")
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]  # Crop to last block_size tokens
                logits, _ = self(idx_cond)
                if logits is None:
                    self.logger.error("Generation failed due to forward pass error")
                    return idx

                logits = logits[:, -1, :]  # (B, C)
                probs = F.softmax(logits, dim=-1)  # (B, C)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            self.logger.info(f"Generation completed, output sequence length: {idx.shape[1]}")
            return idx

        except Exception as e:
            self.logger.error(f"Error in generation: {str(e)}")
            return idx  # Return current idx as fallback
