import hydra
from omegaconf import DictConfig, OmegaConf
import warnings

warnings.filterwarnings("ignore")

from src.etc.logger import CustomLogger, logging
from src.workflows import TrainerPipeline
from data import get_file_path

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    logger = CustomLogger(
        log_name='Main',
        log_level=logging.DEBUG,
        log_dir='logs',
        log_filename="runs.log",
    ).get_logger()

    logger.info(
        f"\nExperiment:\n${OmegaConf.to_yaml(cfg.experiment)}"
        f"\nModel:\n${OmegaConf.to_yaml(cfg.model)}"
        f"\nTokenizer:\n${OmegaConf.to_yaml(cfg.tokenizer)}"
        f"\nData:\n${OmegaConf.to_yaml(cfg.data)}"
        f"\nTraining:\n${OmegaConf.to_yaml(cfg.training)}"
        f"\nLogging:\n${OmegaConf.to_yaml(cfg.logging)}"
        f"\nmlops:\n${OmegaConf.to_yaml(cfg.mlops)}"
    )

    file_path = get_file_path()
    trainer_pipeline = TrainerPipeline(file_path)
    trainer_pipeline.run()


if __name__ == "__main__":
    main()