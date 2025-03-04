import torch

from src.impl.model_ops import generate_text
from src.interface.step import Step
from src.impl.config import Config
from src.impl.eval import estimate_loss
from src.impl.model import GPT2
from src.impl.trainer import Trainer

class ModelCreateStep(Step):
    """
    Step that initializes GPT2 or loads it from a checkpoint (depending on your logic).
    Returns the model and tokenizer (or maybe just the model).
    """
    def __init__(self, config: Config = Config()):
        self.config = config

    def run(self, input_data=None):
        if input_data is None:
            raise ValueError("ModelTrainStep requires input_data")

        tokenizer, test, train = input_data

        model = GPT2(
            tokenizer=tokenizer,
            max_seq_len=self.config.block_size,
            d_model=self.config.d_model,
            num_heads=self.config.n_heads,
            num_layers=self.config.n_layers,
            dropout=self.config.dropout
        )

        return model, tokenizer, test, train


class ModelTrainStep(Step):
    """
    Step that trains the GPT2 model on the given data (and/or config).
    Expects (model, tokenizer) from the previous step, and maybe also training data.
    Returns the trained model (and tokenizer).
    """
    def __init__(self, config: Config = Config(), optimizer=None):
        self.config = config
        self.optimizer = optimizer

    def run(self, input_data=None):
        """
        input_data might be:
          (model, tokenizer, X_train, X_test, y_train, y_test) if you're combining steps
          or you can store data in some global format.
        """
        if input_data is None:
            raise ValueError("ModelTrainStep requires input_data")

        model, tokenizer, test, train = input_data

        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        # Create a Trainer
        trainer = Trainer(
            model=model,
            optimizer=self.optimizer,
            load_exiting_model=False,
            save_on_steps=True,
            device=self.config.device
        )

        trainer.train(train)

        return model, tokenizer, test


class ModelEvaluateStep(Step):
    """
    Step that evaluates the trained model (e.g., on validation data).
    Expects (model, tokenizer) plus maybe a dataset, returns metrics or final results.
    """
    def __init__(self):
        pass

    def run(self, input_data=None):
        if input_data is None:
            raise ValueError("ModelEvaluateStep requires input_data")

        model, tokenizer, test = input_data
        metrics = estimate_loss(model, test)
        print("Evaluation Metrics:", metrics)

        return metrics


class ModelInferenceStep(Step):
    """
    A pipeline Step that performs text generation (inference) with the GPT2 model.
    Expects input_data to be (tokenizer, model, prompt_text).
    Returns the generated text as a string.
    """

    def __init__(self, max_new_tokens: int = 200):
        """
        :param max_new_tokens: How many new tokens to generate for each prompt.
        """
        self.max_new_tokens = max_new_tokens

    def run(self, input_data=None):
        """
        :param input_data: A tuple (tokenizer, model, prompt)
           - tokenizer: from `impl.tokenizer.Tokenizer`
           - model: your GPT2 (trained, loaded)
           - prompt: a string or list of strings for generation
        :return: The generated text (string).
        """
        if not input_data or len(input_data) < 3:
            raise ValueError(
                "InferenceStep expects input_data=(tokenizer, model, prompt_text)."
            )

        prompt, tokenizer, model = input_data

        generated = generate_text(
            tokenizer=tokenizer,
            model=model,
            data=prompt,
            max_new_tokens=self.max_new_tokens
        )

        return generated