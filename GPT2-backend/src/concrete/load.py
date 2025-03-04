from click import prompt
from torch.optim import AdamW

from src.impl.config import Config
from src.impl.model import GPT2
from src.impl.tokenizer import Tokenizer
from src.impl.utils import load_checkpoint
from src.interface import Step


class LoadTokenizerStep(Step):
    """
    A pipeline Step that loads (or initializes) a Tokenizer object.
    Returns the tokenizer.
    """
    def __init__(self, vocab_file: str):
        """
        :param vocab_file: Path to the file containing vocabulary or whatever
                           the Tokenizer needs.
        """
        self.vocab_file = vocab_file

    def run(self, input_data=None):
        """
        Expects no input_data. Returns a Tokenizer instance.
        """
        prompt = None

        if input_data is not None:
            prompt = input_data

        tokenizer = Tokenizer(self.vocab_file)
        return prompt, tokenizer


class LoadModelStep(Step):
    """
    Loads or initializes the GPT2 model (and an optimizer) from a checkpoint if available.
    Expects a tokenizer from the previous step in input_data.
    Returns (model, optimizer).
    """
    def __init__(self, config: Config = None):
        """
        :param config: an optional config. If None, uses default.
        """
        self.config = config if config else Config()

    def run(self, input_data=None):
        """
        :param input_data: the tokenizer from the previous step
        :return: (model, optimizer)
        """

        prompt, tokenizer = input_data  # We expect the tokenizer as input_data
        model = GPT2(
            tokenizer=tokenizer,
            d_model=self.config.d_model,
            max_seq_len=self.config.block_size,
            num_heads=self.config.n_heads,
            dropout=self.config.dropout,
            num_layers=self.config.n_layers
        )

        # Create optimizer
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)

        # Try to load checkpoint if it exists
        epoch, loss = load_checkpoint(model, optimizer)
        if epoch == 0:
            print("No checkpoint found. Model is newly initialized.")
        else:
            print(f"Checkpoint loaded. Epoch={epoch}, Loss={loss}")

        return prompt, tokenizer, model
