from click import prompt
from torch.optim import AdamW

from src.impl.config import Config
from src.impl.utils import load_checkpoint
from src.interface import Step
from src.impl.model_ops import setup_model, setup_trainer, setup_optimizer, setup_tokenizer

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
        tokenizer = setup_tokenizer(self.vocab_file)

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
        model = setup_model(tokenizer, self.config)
        optimizer = setup_optimizer(model, self.config)

        # Try to load checkpoint if it exists
        epoch, loss = load_checkpoint(model, optimizer)
        if epoch == 0:
            print("No checkpoint found. Model is newly initialized.")
        else:
            print(f"Checkpoint loaded. Epoch={epoch}, Loss={loss}")

        return prompt, tokenizer, model
