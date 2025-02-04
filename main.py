from functools import lru_cache
from src import Config
from src import GPT2, Tokenizer, Optimizer
from src import Trainer


data = []

@lru_cache()
def load_data():
    global data
    with open("src/data/ClimateChangeAnalysis.txt", "rt") as file:
        data = file.read()


load_data()
tokenizer = Tokenizer(data)
encoded_data = tokenizer.encode(data)

model = GPT2(
    vocab_size  = tokenizer.get_vocab_size(),
    d_model     = Config.d_model,
    max_seq_len = Config.block_size,
    batch_size  = Config.batch_size,
    num_heads   = Config.n_heads,
    dropout     = Config.dropout,
    num_layers  = Config.n_layers
)

optimizer = Optimizer(
    params      = model.parameters(),
    lr          = Config.learning_rate
)

trainer = Trainer(
    model           = model,
    optimizer       = optimizer,
    save_on_steps   = True
)

trainer.train(encoded_data)