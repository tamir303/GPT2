from functools import lru_cache

import torch

from src.tokenizer import Tokenizer
from src.embedding import EmbeddingTable, PositionalEncoding
from src.config import Config

data = []

@lru_cache()
def load_data():
    global data
    with open("src/data/ClimateChangeAnalysis.txt", "rt") as file:
        data = file.read()

load_data()

tokenizer = Tokenizer(data)
tokens_idx = tokenizer.encode(data)

emb = EmbeddingTable(vocab_size=tokenizer.get_vocab_size(), d_model=Config.d_model)
idx = emb(tokens_idx[:Config.block_size])
pe = PositionalEncoding(d_model=Config.d_model, max_seq_len=Config.block_size)
idx = pe(idx)