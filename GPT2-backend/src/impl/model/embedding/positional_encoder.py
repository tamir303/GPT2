import torch
import torch.nn as nn

from src.etc.logger import CustomLogger, logging

pos_encoding_logger = CustomLogger(
    log_name='PositionalEncoding',
    log_level=logging.DEBUG,
    log_dir='app_logs',
    log_filename='positional_encoding.log'
).get_logger()


class PositionalEncoding(nn.Module):
    n = 10000

    def __init__(self,
                 d_model: int,
                 block_size: int):
        super().__init__()
        try:
            pe = torch.zeros(block_size, d_model)  # (T, C)
            positions = torch.arange(block_size).unsqueeze(1)
            div_term = torch.pow(self.n, -torch.arange(start=0, end=d_model, step=2) / d_model)

            pe[:, 0::2] = torch.sin(positions * div_term)
            pe[:, 1::2] = torch.cos(positions * div_term)

            self.register_buffer('pe', pe)
            pos_encoding_logger.info(f"Initialized PositionalEncoding with d_model={d_model}, max_seq_len={block_size}")
        except Exception as e:
            pos_encoding_logger.error(f"Failed to initialize PositionalEncoding: {str(e)}")
            raise

    def forward(self, x: torch.Tensor):
        try:
            seq_len = x.size(dim=1)
            if seq_len > self.pe.size(0):
                pos_encoding_logger.warning(
                    f"Input sequence length {seq_len} exceeds max_seq_len {self.pe.size(0)}, truncating"
                )
                seq_len = self.pe.size(0)
            output = x + self.pe[:seq_len]
            pos_encoding_logger.debug(f"Forward pass completed: input shape={x.shape}, output shape={output.shape}")
            return output
        except Exception as e:
            pos_encoding_logger.error(f"Error in forward pass with input shape={x.shape}: {str(e)}")
            return x  # Return input as fallback to avoid breaking downstream computation