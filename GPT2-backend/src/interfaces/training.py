from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.optim import Optimizer


class ITraining(ABC):
    """Abstract base class for training logic."""

    def __init__(self, model: nn.Module, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer

    @abstractmethod
    def train(self, data: torch.Tensor) -> None:
        """
        Train the model on the given data.

        Args:
            data (torch.Tensor): Training data
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up resources (e.g., close loggers, writers).
        """
        pass