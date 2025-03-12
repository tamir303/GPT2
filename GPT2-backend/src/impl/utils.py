import torch
from torch import nn
import logging

from src.etc.logger import CustomLogger
from src.etc.config import Config

# Initialize logger for utils
utils_logger = CustomLogger(
    log_name='Utils',
    log_level=logging.DEBUG,
    log_dir='app_logs',
    log_filename='utils.log'
).get_logger()


def split_train_test(data: torch.Tensor, split: float = 0.9) -> [str, str]:
    """
    Split data into training and validation sets.

    Args:
        data (torch.Tensor): Input data tensor
        split (float): Fraction of data to use for training (default: 0.9)

    Returns:
        tuple: (train_data, val_data) tensors
    """
    try:
        n = int(split * len(data))
        if n <= 0 or n >= len(data):
            raise ValueError(f"Invalid split size: {n} (data length: {len(data)}, split: {split})")
        train_data = data[:n]
        val_data = data[n:]
        utils_logger.debug(f"Data split: train shape={train_data.shape}, val shape={val_data.shape}")
        return train_data, val_data
    except Exception as e:
        utils_logger.error(f"Error splitting data: {str(e)}")
        return data, None  # Return full data as train and empty tensor as val as fallback


def get_batch(data: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    """
    Generate a batch of input-target pairs from the data.

    Args:
        data (torch.Tensor): Input data tensor

    Returns:
        tuple: (x, y) tensors where x is input and y is target
    """
    try:
        if len(data) <= Config.block_size:
            raise ValueError(f"Data length {len(data)} is too short for block_size {Config.block_size}")
        ix = torch.randint(len(data) - Config.block_size, (Config.batch_size,))
        x = torch.stack([data[i:i + Config.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + Config.block_size + 1] for i in ix])
        x, y = x.to(Config.device), y.to(Config.device)
        utils_logger.debug(f"Batch generated: x shape={x.shape}, y shape={y.shape}, device={x.device}")
        return x, y
    except Exception as e:
        utils_logger.error(f"Error generating batch: {str(e)}")
        return torch.tensor([]), torch.tensor([])  # Return empty tensors as fallback


def load_checkpoint(model: nn.Module, optimizer, filename: str) -> tuple[int, float]:
    """
    Load a model checkpoint from a file.

    Args:
        model (nn.Module): Model to load state into
        optimizer: Optimizer to load state into
        filename (str): Path to checkpoint file

    Returns:
        tuple: (epoch, loss) from the checkpoint
    """
    try:
        checkpoint = torch.load(filename)
        if checkpoint is None or not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint format")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        utils_logger.info(f"Checkpoint loaded from {filename}. Resuming from epoch {epoch}, loss {loss:.4f}")
        return epoch, loss
    except FileNotFoundError:
        utils_logger.warning(f"Checkpoint file {filename} not found, starting from scratch")
        return 0, float("inf")
    except Exception as e:
        utils_logger.error(f"Error loading checkpoint from {filename}: {str(e)}")
        return 0, float("inf")


def save_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float, filename: str):
    """
    Save a model checkpoint to a file.

    Args:
        model (nn.Module): Model to save
        optimizer: Optimizer to save
        epoch (int): Current epoch
        loss (float): Current loss
        filename (str): Path to save checkpoint
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filename)
        utils_logger.info(f"Checkpoint saved at {filename} for epoch {epoch}, loss {loss:.4f}")
    except Exception as e:
        utils_logger.error(f"Error saving checkpoint to {filename}: {str(e)}")

