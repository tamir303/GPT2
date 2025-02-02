from functools import lru_cache
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

emb = EmbeddingTable(tokenizer.get_vocab_size(), Config.d_model)
tokens_idx = tokenizer.encode(data)
idx = emb(tokens_idx)

pe = PositionalEncoding(Config.d_model, Config.block_size)
idx = pe(idx)
print(idx)