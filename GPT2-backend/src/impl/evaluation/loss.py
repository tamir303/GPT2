from typing import Dict

import torch
from torch import nn
import logging

from src.etc.logger import CustomLogger
from src.etc.config import Config
from src.impl.utils import get_batch, split_train_test

# Initialize logger for evaluation
eval_logger = CustomLogger(
    log_name='Evaluation',
    log_level=logging.DEBUG,
    log_dir='app_logs',
    log_filename='evaluation.log'
).get_logger()


@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor) -> Dict:
    """
    Estimate the loss on train and validation splits of the data.

    Args:
        model (nn.Module): The model to evaluate
        data (torch.Tensor): The input data tensor

    Returns:
        Dict: Dictionary containing mean loss for 'train' and 'validation' splits
    """
    out = {}
    try:
        model.eval()
        eval_logger.info("Starting loss estimation")
    except Exception as e:
        eval_logger.error(f"Failed to set model to eval mode: {str(e)}")
        return out  # Return empty dict as fallback

    try:
        train, val = split_train_test(data)
        eval_logger.debug(f"Data split: train shape={train.shape}, val shape={val.shape}")
    except Exception as e:
        eval_logger.error(f"Error splitting data: {str(e)}")
        return out

    for split_type, split_data in [("train", train), ("validation", val)]:
        losses = torch.zeros(Config.eval_iters)
        for iter in range(Config.eval_iters):
            try:
                x, y = get_batch(split_data)
                _, loss = model(x, y)
                if loss is None:
                    eval_logger.warning(f"Loss is None for {split_type} split at iteration {iter}")
                    losses[iter] = 0.0  # Default to 0 if loss is None
                else:
                    losses[iter] = loss.item()
            except Exception as e:
                eval_logger.error(f"Error computing loss for {split_type} split at iteration {iter}: {str(e)}")
                losses[iter] = 0.0  # Default to 0 on error to continue evaluation

        try:
            mean_loss = losses.mean()
            out[split_type] = mean_loss
            eval_logger.info(f"{split_type.capitalize()} loss: {mean_loss:.4f}")
        except Exception as e:
            eval_logger.error(f"Error calculating mean loss for {split_type}: {str(e)}")
            out[split_type] = 0.0  # Default to 0 if mean calculation fails

    try:
        model.train()
        eval_logger.info("Loss estimation completed, model set back to train mode")
    except Exception as e:
        eval_logger.error(f"Failed to set model back to train mode: {str(e)}")

    return out

