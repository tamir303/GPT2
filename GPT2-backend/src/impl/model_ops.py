import mlflow
from typing import List, Union
from src.impl.trainer import Trainer
from src.impl.eval import estimate_loss
from src.impl.config import Config
from src.impl.logger import get_logger

logger = get_logger()

def get_info() -> dict:
    return {
        "title": "Transformer Model API",
        "description": "Transformer Model API",
        "version": "1.0.0",
        "device": Config.device,
        "d_model": Config.d_model,
        "block_size": Config.block_size,
        "n_heads": Config.n_heads,
        "n_layers": Config.n_layers,
        "dropout": Config.dropout,
        "learning_rate": Config.learning_rate,
        "eval_iters": Config.eval_iters,
        "eval_interval": Config.eval_interval,
    }

def start_experiment(experiment_name: str, config: Config) -> mlflow.ActiveRun:
    """
    Starts an MLflow experiment run and logs configuration parameters.
    Returns the active mlflow run, which you can keep to reference later.
    """
    mlflow.set_experiment(experiment_name)
    mlflow_run = mlflow.start_run()
    logger.info(f"MLflow run started: {mlflow_run.info.run_id}")

    # Log config params to MLflow
    mlflow.log_params({
        "d_model": config.d_model,
        "block_size": config.block_size,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "learning_rate": config.learning_rate,
        "dropout": config.dropout,
        "device": config.device
    })
    return mlflow_run


def end_experiment() -> None:
    """
    Ends the current MLflow experiment run.
    """
    mlflow.end_run()
    logger.info("MLflow run ended.")


def train_model(
    tokenizer,
    model,
    trainer: Trainer,
    data: Union[List[str], str]
) -> None:
    """
    Trains the model on the provided data.
    - tokenizer, model, trainer: from setup_model
    - data: text or list of text prompts
    """
    if not trainer or not tokenizer:
        raise RuntimeError("Trainer or tokenizer not initialized. Call setup_model() first.")

    logger.info("Starting model training...")

    # Tokenize data
    encoded_data = tokenizer.encode(data)

    # Actually train the model
    trainer.train(encoded_data)

    # After training, we can log final metrics
    metrics = estimate_loss(model, encoded_data)
    mlflow.log_metrics(metrics)
    logger.info(f"Training metrics: {metrics}")


def generate_text(
    tokenizer,
    model,
    data: Union[List[str], str],
    max_new_tokens: int = 2000
) -> str:
    """
    Generates text based on the provided prompt(s).
    - tokenizer, model: from setup_model
    - data: input text (string or list of strings)
    """
    if not model or not tokenizer:
        raise RuntimeError("Model or tokenizer not initialized. Call setup_model() first.")

    logger.info("Generating text from prompt...")
    # Encode the prompt(s)
    context = tokenizer.encode(data)

    # Use GPT2's generate method
    generated_ids = model.generate(context, max_new_tokens=max_new_tokens)
    result_str = tokenizer.decode(generated_ids[0].tolist())

    # Log the generated text to MLflow
    mlflow.log_text(result_str, artifact_file="generated_text.txt")
    logger.info("Text generation complete.")
    return result_str
