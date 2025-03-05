from typing import Union, List

import torch


class LetterTokenizer:
    def __init__(self, *args):
        # Define the vocabulary: a-z, space, and some punctuation
        self.vocab = "abcdefghijklmnopqrstuvwxyz .,!?"
        self.vocab_size = len(self.vocab)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        text = text.lower()  # Convert to lowercase for consistency
        token_ids = []
        for char in text:
            # Use '?' as a fallback for unknown characters
            token_id = self.char_to_id.get(char, self.char_to_id['?'])
            token_ids.append(token_id)
        return torch.tensor([token_ids])

    def decode(self, token_ids: list) -> str:
        try:
            chars = []
            for token_id in token_ids:
                # Check if token_id is valid, otherwise use '?'
                if isinstance(token_id, int) and 0 <= token_id < self.vocab_size:
                    char = self.id_to_char[token_id]
                else:
                    char = '?'
                chars.append(char)
            return ''.join(chars)
        except Exception as e:
            print(f"Decoding error: {e}")
            return ""

    def get_vocab_size(self) -> int:
        return self.vocab_size