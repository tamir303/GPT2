from typing import List, Union

import mlflow
from torch.optim import AdamW as Optimizer

from src.tokenizer import Tokenizer
from src.model import GPT2
from src.trainer import Trainer

from src.eval import estimate_loss
from src.utils import load_checkpoint
from src.config import Config
from src.logger import get_logger

logger = get_logger()


class ModelManager:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.trainer = None

    def load_model(self, data: Union[List[str], str], file_path: str) -> None:
        """Initializes the tokenizer, model, optimizer, and trainer using the provided data."""
        try:
            logger.info("Initializing tokenizer and model components...")
            self.tokenizer = Tokenizer(file_path)

            self.model = GPT2(
                tokenizer=self.tokenizer,
                d_model=self.config.d_model,
                max_seq_len=self.config.block_size,
                num_heads=self.config.n_heads,
                dropout=self.config.dropout,
                num_layers=self.config.n_layers
            )

            self.optimizer = Optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate
            )

            self.trainer = Trainer(
                model=self.model,
                optimizer=self.optimizer,
                save_on_steps=True,
                load_exiting_model=True,
                device=self.config.device
            )

            epoch, loss = load_checkpoint(self.model, self.optimizer)
            if epoch == 0:
                logger.info("No Model found...\n training new one!.")
                self.train_model(data)
            else:
                logger.info("Model loaded successfully.")

            self.__start_experiment("CustomGPT2Experiment")

        except Exception as e:
            logger.exception("Error while loading model components.")
            raise e

    def train_model(self, data: Union[List[str], str]) -> None:
        """Trains the model on the provided data."""
        if not self.trainer or not self.tokenizer:
            raise RuntimeError("ModelManager not initialized. Call load_model() first.")

        encoded_data = self.tokenizer.encode(data)
        logger.info("Starting model training...")
        self.trainer.train(encoded_data)

        # Here, one might log final training metrics. For example:
        metrics = estimate_loss(self.model, encoded_data)
        mlflow.log_metrics(metrics)
        logger.info(f"Training metrics: {metrics}")

    def generate_text(self, data: Union[List[str], str], max_new_tokens: int = 2000) -> str:
        """
        Generates text based on the provided prompt.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("ModelManager not initialized. Call load_model() first.")
        context = self.tokenizer.encode(data)
        logger.info("Generating text from prompt...")

        generated_ids = self.model.generate(context, max_new_tokens=max_new_tokens)
        result = self.tokenizer.decode(generated_ids[0].tolist())
        logger.info("Text generation complete.")

        mlflow.log_text("".join(result), artifact_file="generated_text.txt")
        return "".join(result)

    def __start_experiment(self, experiment_name: str = "DefaultExperiment") -> None:
        """
        Starts an MLflow experiment run and logs configuration parameters.
        """
        mlflow.set_experiment(experiment_name)
        self.mlflow_run = mlflow.start_run()
        logger.info(f"MLflow run started: {self.mlflow_run.info.run_id}")
        # Log configuration parameters
        mlflow.log_params({
            "d_model": self.config.d_model,
            "block_size": self.config.block_size,
            "n_heads": self.config.n_heads,
            "n_layers": self.config.n_layers,
            "learning_rate": self.config.learning_rate,
            "dropout": self.config.dropout,
            "device": self.config.device
        })

    def end_experiment(self) -> None:
        """
        Ends the current MLflow experiment run.
        """
        mlflow.end_run()
        logger.info("MLflow run ended.")
