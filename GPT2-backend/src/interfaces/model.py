from abc import ABC, abstractmethod
import torch
from torch import nn

class IModel(ABC, nn.Module):
    """Abstract base class for models."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor
            targets (torch.Tensor, optional): Target tensor for supervised learning

        Returns:
            tuple: (logits, loss) where loss is None if targets is None
        """
        pass

    @abstractmethod
    def generate(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate new tokens based on input.

        Args:
            x (torch.Tensor): Input tensor
            max_new_tokens (int): Number of new tokens to generate

        Returns:
            torch.Tensor: Generated sequence
        """
        pass