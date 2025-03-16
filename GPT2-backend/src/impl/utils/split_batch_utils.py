import torch
import logging
from typing import Tuple

from src.etc.logger import CustomLogger
from src.etc.config import Config

# Initialize logger for utils
utils_logger = CustomLogger(
    log_name='split_batch_utils',
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
        Config.log_debug_activate and utils_logger.debug("Data split: train shape=%s, val shape=%s", train_data.shape, val_data.shape)
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
        Config.log_debug_activate and utils_logger.debug("Batch generated: x shape=%s, y shape=%s, device=%s", x.shape, y.shape, x.device)
        return x, y
    except Exception as e:
        utils_logger.error("Error generating batch: %s", str(e))
        return torch.tensor([]), torch.tensor([])  # Fallback: return empty tensors