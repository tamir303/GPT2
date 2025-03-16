import torch
import torch.nn as nn
import logging

from src.etc.logger import CustomLogger
from src.etc.config import Config

# Initialize logger for EmbeddingTable
embedding_logger = CustomLogger(
    log_name='EmbeddingTable',
    log_level=logging.DEBUG,
    log_dir='app_logs',
    log_filename='embedding_table.log'
).get_logger()


class EmbeddingTable(nn.Module):
    def __init__(self,
                 d_model: int,
                 vocab_size: int):
        super().__init__()
        try:
            self.et = nn.Embedding(vocab_size, d_model)  # (V, C)
            embedding_logger.info(f"Initialized EmbeddingTable with vocab_size={vocab_size}, d_model={d_model}")
        except Exception as e:
            embedding_logger.error(f"Failed to initialize EmbeddingTable: {str(e)}")
            raise

    def forward(self, x: torch.Tensor):
        try:
            output = self.et(x)  # (B, T, C)
            Config.log_debug_activate and embedding_logger.debug(f"Forward pass completed: input shape={x.shape}, output shape={output.shape}")
            return output
        except Exception as e:
            embedding_logger.error(f"Error in forward pass with input shape={x.shape}: {str(e)}")
            return None  # Return None as a fallback