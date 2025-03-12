from abc import ABC, abstractmethod
from typing import Dict
import torch
from torch import nn


class IEvaluation(ABC):
    """Abstract base class for evaluation logic."""

    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def estimate_loss(self, data: torch.Tensor) -> Dict[str, float]:
        """
        Estimate loss on the given data.

        Args:
            data (torch.Tensor): Data to evaluate on

        Returns:
            Dict[str, float]: Dictionary of split names to loss values
        """
        pass

    @abstractmethod
    def evaluate_metrics(self, data: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate additional metrics (e.g., perplexity) on the data.

        Args:
            data (torch.Tensor): Data to evaluate on

        Returns:
            Dict[str, float]: Dictionary of metric names to values
        """
        pass