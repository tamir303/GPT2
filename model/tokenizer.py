from torchtext.data.utils import get_tokenizer
from typing import List, Union, LiteralString
import torch

class Tokenizer:
    def __init__(self, x: Union[str, List[str]]):
        self.tokenizer = get_tokenizer("basic_english")
        self.tokens = self.tokenizer(x)
        self.stoi = { t: i for i, t in enumerate(self.tokens) }
        self.itos = { i: t for i, t in enumerate(self.tokens)}

    def encode(self, tokens: Union[LiteralString, List[LiteralString]]) -> torch.Tensor:
        return torch.Tensor([ self.stoi[token] for token in tokens ])

    def decode(self, token_ids: torch.Tensor) -> Union[LiteralString, List[LiteralString]]:
        return [ self.itos[token_id] for token_id in token_ids ]