from typing import Union

import torch
from torch import nn

from src.impl.config import Config


def split_train_test(data: Union[str, list, torch.Tensor], split=0.9) -> [torch.Tensor, torch.Tensor]:
    # Train and test splits
    n = int(split * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def get_batch(data: Union[str, list, torch.Tensor]) -> [torch.Tensor, torch.Tensor]:
    # generate a small batch of data of inputs x and targets y
    if isinstance(data, str):
        data = [ord(ch) for ch in data]

    if isinstance(data, list):
        data = torch.tensor(data, dtype=torch.long)

    data_length = data.size(0)
    if data_length <= Config.block_size:
        raise ValueError("Data is too short for the block_size.")

    ix = torch.randint(
        low=0,
        high=data_length - Config.block_size,
        size=(Config.batch_size,)
    )

    x = torch.stack([data[i : i + Config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + Config.block_size + 1] for i in ix])

    x = x.to(Config.device)
    y = y.to(Config.device)

    return x, y


def load_checkpoint(model: nn.Module, optimizer, filename=Config.filename):
    try:
        checkpoint = torch.load(filename)
        if checkpoint is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Checkpoint loaded. Resuming from epoch {epoch}, loss {loss:.4f}.")
            return epoch, loss
    except Exception as e:
        print(f"Warning: {e}")
        return 0, float("inf")


def save_checkpoint(model, optimizer, epoch, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, Config.filename)
    print(f"\nCheckpoint saved at epoch {epoch}.")
