from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    batch_size: int = 512
