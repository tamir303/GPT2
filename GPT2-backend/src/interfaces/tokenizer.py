from abc import ABC, abstractmethod
import torch

class ITokenizer(ABC):
    """Abstract base class for tokenizers."""

    def __init__(self):
        pass

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text into a tensor of token IDs.

        Args:
            text (str): Input text

        Returns:
            torch.Tensor: Encoded token IDs
        """
        pass

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> str:
        """
        Decode token IDs into text.

        Args:
            tokens (torch.Tensor): Tensor of token IDs

        Returns:
            str: Decoded text
        """
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        """
        Get the size of the tokenizer's vocabulary.

        Returns:
            int: Vocabulary size
        """
        pass