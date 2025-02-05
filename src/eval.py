from typing import Dict

import torch
from torch import nn

from src.config import Config
from src.utils import get_batch, split_train_test


@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor) -> Dict:
    out = {}
    model.eval()

    train, val = split_train_test(data)
    for split_type, split_data in [("train", train), ("validation", val)]:
        losses = torch.zeros(Config.eval_iters)
        for iter in range(Config.eval_iters):
            # Get random batch of X, y
            x, y = get_batch(split_data)
            _, loss = model(x, y)
            losses[iter] = loss.item()

        out[split_type] = losses.mean()

    model.train()
    return out
