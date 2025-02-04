from functools import lru_cache
from src import split_train_test, get_batch, Config, GPT2, Tokenizer

data = []

@lru_cache()
def load_data():
    global data
    with open("src/data/ClimateChangeAnalysis.txt", "rt") as file:
        data = file.read()


load_data()
tokenizer = Tokenizer(data)

model = GPT2(
    vocab_size  = tokenizer.get_vocab_size(),
    d_model     = Config.d_model,
    max_seq_len = Config.block_size,
    batch_size  = Config.batch_size,
    num_heads   = Config.n_heads,
    dropout     = Config.dropout,
    num_layers  = Config.n_layers
)