import torch
from src.config import Config

def split_train_test(data: torch.Tensor, split=0.9) -> [torch.Tensor, torch.Tensor]:
    # Train and test splits
    n = int(split * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data

def get_batch(data: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - Config.block_size, (Config.batch_size, ))
    x = torch.stack([data[i:i + Config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + Config.block_size + 1] for i in ix])
    x, y = x.to(Config.device), y.to(Config.device)
    return x, y