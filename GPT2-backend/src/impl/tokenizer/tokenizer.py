import logging
import os
import multiprocessing
from typing import List, Union

import sentencepiece as spm
import torch

from src.etc.logger import CustomLogger
from src.interfaces.tokenizer import ITokenizer
from src.etc.config import Config


class Tokenizer(ITokenizer):
    """Efficient SentencePiece Tokenizer with multiprocessing training."""

    __MODEL_DIR = "tokenizer"
    __MODEL_FILE = os.path.join(__MODEL_DIR, "tok400.model")

    __options = {
        "input_format": "text",
        "model_prefix": os.path.join(__MODEL_DIR, "tok400"),
        "model_type": "bpe",
        "vocab_size": 10000,
        "normalization_rule_name": "identity",
        "remove_extra_whitespaces": False,
        "input_sentence_size": 200000000,
        "max_sentence_length": 4192,
        "seed_sentencepiece_size": 1000000,
        "shuffle_input_sentence": True,
        "character_coverage": 0.99995,
        "byte_fallback": True,
        "split_digits": True,
        "split_by_unicode_script": True,
        "split_by_whitespace": True,
        "split_by_number": True,
        "max_sentencepiece_length": 16,
        "add_dummy_prefix": True,
        "allow_whitespace_only_pieces": True,
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": -1,
        "num_threads": os.cpu_count(),
    }

    def __init__(self, file_path: str):
        super().__init__()

        # Setup logger
        self.logger = CustomLogger(
            log_name="Tokenizer",
            log_level=logging.DEBUG,
            log_dir="app_logs",
            log_filename="tokenizer.log",
        ).get_logger()

        self.file_path = os.path.abspath(file_path)
        self.sp = spm.SentencePieceProcessor()

        # Ensure tokenizer directory exists
        os.makedirs(self.__MODEL_DIR, exist_ok=True)

        # Try to load existing model or train a new one
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        """Load or train the SentencePiece model."""
        try:
            if os.path.exists(self.__MODEL_FILE):
                self.sp.Load(self.__MODEL_FILE)
                self.logger.info("Loaded existing SentencePiece model.")
            else:
                self.logger.warning("SentencePiece model not found, starting training...")
                self.__options["input"] = self.file_path
                self.train_model()
        except Exception as e:
            self.logger.error(f"Error loading SentencePiece model: {str(e)}")
            self.train_model()

    def train_model(self):
        """Train SentencePiece model using multiprocessing."""
        self.logger.info("Training SentencePiece model using multiprocessing...")

        system_workers = min(Config.num_workers, os.cpu_count())
        num_workers = max(1, system_workers // 2)  # Avoid freezing the system

        # Ensure tokenizer directory exists before training
        os.makedirs(self.__MODEL_DIR, exist_ok=True)

        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.apply(spm.SentencePieceTrainer.Train, kwds=self.__options)

        self.sp.Load(self.__MODEL_FILE)  # Load the newly trained model
        self.logger.info("Finished training and loading SentencePiece model.")

    def encode(self, raw: Union[str, List[str]]) -> torch.Tensor:
        """Tokenize input text and return a tensor of token IDs."""
        self.logger.debug("Encoding input text...")
        return torch.tensor(self.sp.Encode(raw))

    def decode(self, token_ids: torch.Tensor) -> Union[str, List[str]]:
        """Convert token IDs back to text."""
        self.logger.debug("Decoding token IDs...")
        return self.sp.Decode(token_ids.tolist())

    def get_vocab_size(self) -> int:
        """Return the vocabulary size of the tokenizer."""
        return self.sp.vocab_size()
