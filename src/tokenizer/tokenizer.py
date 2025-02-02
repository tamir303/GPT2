from torchtext.data.utils import get_tokenizer
from typing import List, Union
import torch

class Tokenizer:
    def __init__(self, x: Union[str, List[str]]):
        self.tokenizer = get_tokenizer("basic_english")
        words = set(self.tokenizer(x))

        self.stoi = { t: i for i, t in enumerate(words) }
        self.itos = { i: t for i, t in enumerate(words) }
        self.encoded_data = self.encode(x)

    def encode(self, input: Union[str, List[str]]) -> torch.Tensor:
        tokens = self.tokenizer(input)
        return torch.tensor([ self.stoi.get(token, -1) for token in tokens ], dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> Union[str, List[str]]:
        return [ " ".join(self.itos.get(token_id, "[INVALID]") for token_id in token_ids.tolist()) ]

    @staticmethod
    def __map_to_lower(tokens: Union[str, List[str]]) -> Union[str, List[str]]:
        return [token.lower() for token in tokens]

    def get_vocab_size(self) -> int:
        return len(self.stoi)