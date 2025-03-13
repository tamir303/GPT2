import torch
from torch import nn
import logging
from typing import Tuple

from src.etc.logger import CustomLogger
from src.etc.config import Config

# Initialize logger for utils
utils_logger = CustomLogger(
    log_name='Utils',
    log_level=logging.DEBUG,
    log_dir='app_logs',
    log_filename='utils.log'
).get_logger()


def split_train_test(data: torch.Tensor, split: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split data into training and validation sets.

    Args:
        data (torch.Tensor): Input data tensor.
        split (float): Fraction of data to use for training (default: 0.9).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (train_data, val_data) tensors.
    """
    try:
        n = int(split * len(data))
        if n <= 0 or n >= len(data):
            raise ValueError("Invalid split size: %d (data length: %d, split: %f)" % (n, len(data), split))
        train_data = data[:n]
        val_data = data[n:]
        utils_logger.debug("Data split: train shape=%s, val shape=%s", train_data.shape, val_data.shape)
        return train_data, val_data
    except Exception as e:
        utils_logger.error("Error splitting data: %s", str(e))
        return data, torch.tensor([])  # Fallback: return full data and an empty tensor


def get_batch(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of input-target pairs from the data using vectorized operations.

    Args:
        data (torch.Tensor): Input data tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (x, y) tensors where x is input and y is target.
    """
    try:
        if len(data) <= Config.block_size:
            raise ValueError("Data length %d is too short for block_size %d" % (len(data), Config.block_size))

        # Create sliding windows of length (block_size + 1)
        windows = data.unfold(0, Config.block_size + 1, 1)  # Shape: (L - block_size, block_size + 1)
        num_windows = windows.size(0)

        # Randomly select batch indices
        batch_indices = torch.randint(0, num_windows, (Config.batch_size,))
        batch_windows = windows[batch_indices]  # Shape: (batch_size, block_size + 1)

        # Split each window into input (x) and target (y)
        x = batch_windows[:, :-1]
        y = batch_windows[:, 1:]

        x, y = x.to(Config.device), y.to(Config.device)
        utils_logger.debug("Batch generated: x shape=%s, y shape=%s, device=%s", x.shape, y.shape, x.device)
        return x, y
    except Exception as e:
        utils_logger.error("Error generating batch: %s", str(e))
        return torch.tensor([]), torch.tensor([])  # Fallback: return empty tensors


def load_checkpoint(model: nn.Module, optimizer, filename: str) -> Tuple[int, float]:
    """
    Load a model checkpoint from a file.

    Args:
        model (nn.Module): Model to load state into.
        optimizer: Optimizer to load state into.
        filename (str): Path to checkpoint file.

    Returns:
        Tuple[int, float]: (epoch, loss) from the checkpoint.
    """
    try:
        checkpoint = torch.load(filename)
        if checkpoint is None or not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint format")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        utils_logger.info("Checkpoint loaded from %s. Resuming from epoch %d, loss %.4f", filename, epoch, loss)
        return epoch, loss
    except FileNotFoundError:
        utils_logger.warning("Checkpoint file %s not found, starting from scratch", filename)
        return 0, float("inf")
    except Exception as e:
        utils_logger.error("Error loading checkpoint from %s: %s", filename, str(e))
        return 0, float("inf")


def save_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float, filename: str):
    """
    Save a model checkpoint to a file.

    Args:
        model (nn.Module): Model to save.
        optimizer: Optimizer to save.
        epoch (int): Current epoch.
        loss (float): Current loss.
        filename (str): Path to save checkpoint.
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filename)
        utils_logger.info("Checkpoint saved at %s for epoch %d, loss %.4f", filename, epoch, loss)
    except Exception as e:
        utils_logger.error("Error saving checkpoint to %s: %s", filename, str(e))
