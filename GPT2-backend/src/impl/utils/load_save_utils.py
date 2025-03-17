import os
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from typing import Tuple
from src.etc.config import Config
from src.etc.logger import CustomLogger
import logging

from src.repo.model_schema import ModelSchema, ModelTrainStatus
from src.repo.model_repo import ModelRepository


class LoadSaveUtilsClass:
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 identifier: str):

        # Initialize logger for utils
        self.utils_logger = CustomLogger(
            log_name='Utils',
            log_level=logging.DEBUG,
            log_dir='app_logs',
            log_filename='utils.log'
        ).get_logger()

        # Create Pydantic ModelSchema
        model_schema_params = {
            "type": Config.name,
            "d_model": Config.d_model,
            "block_size": Config.block_size,
            "n_heads": Config.n_heads,
            "n_layers": Config.n_layers,
            "dropout": Config.dropout,
        }

        self.model_schema = ModelSchema(**model_schema_params)

        self.model = model
        self.optimizer = optimizer
        self.identifier = identifier

        try:
            self.model_repo = ModelRepository()
        except Exception as e:
            self.utils_logger.error("Error initializing ModelRepository: %s", str(e))
            self.model_repo = None

    def __save_checkpoint_temp(self, epoch: int, loss: float) -> str:
        """
        Save a model checkpoint to a temporary file.

        Args:
            epoch (int): Current epoch.
            loss (float): Current loss.

        Returns:
            str: Full path to the temporary checkpoint file.
        """
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }

            # Create a temporary directory for checkpoints
            temp_dir = Path("temp_checkpoints")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / f"temp_checkpoint_{self.identifier}.pth"
            torch.save(checkpoint, temp_file)

            if Config.log_debug_activate:
                self.utils_logger.info(
                    "Temporary checkpoint saved at %s for epoch %d, loss %.4f", temp_file, epoch, loss
                )

            return str(temp_file)
        except Exception as e:
            self.utils_logger.error("Error saving temporary checkpoint: %s", str(e))
            raise

    def load_checkpoint(self) -> Tuple[int, float, str | None]:
        """
        Load a model checkpoint from MongoDB.

        Returns:
            Tuple[int, float, str]: (epoch, loss, identifier) from the checkpoint.

        Raises:
            ValueError: If MongoDB ID is not found or invalid.
        """

        try:
            # Load from MongoDB directly
            model_schema, train_status_schema, checkpoint_io = self.model_repo.load(self.identifier)
            self.utils_logger.info(f"Loading model... \n schema: {model_schema}, train status: {train_status_schema}")

            # Load checkpoint from IO stream
            checkpoint = torch.load(checkpoint_io)

            if checkpoint is None or not isinstance(checkpoint, dict):
                raise ValueError("Invalid checkpoint format from MongoDB")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            Config.log_debug_activate and self.utils_logger.info(
                "Checkpoint loaded from MongoDB ID %s. Resuming from epoch %d, loss %.4f",
                self.identifier, epoch, loss)

            return epoch, loss, self.identifier
        except ValueError as ve:
            self.utils_logger.error("Error loading checkpoint from MongoDB ID %s: %s", self.identifier, str(ve))
            return 0, float("inf"), None
        except Exception as e:
            self.utils_logger.error("Unexpected error loading from MongoDB ID %s: %s", self.identifier, str(e))
            return 0, float("inf"), None

    def save_checkpoint(self, epoch: int, loss: dict, accuracy: float) -> str:
        """
        Save a model checkpoint to MongoDB only.

        Args:
            epoch (int): Current epoch.
            loss (dict): Dictionary containing loss values.
            accuracy (float): Current accuracy.

        Returns:
            str: MongoDB ModelEntry ID.

        Raises:
            ValueError: If schema is missing for MongoDB.
        """

        # Create ModelTrainStatus
        model_train_status_params = {
            "current_epoch": epoch,
            "val_loss": loss["validation"],
            "train_loss": loss["train"],
            "accuracy": accuracy
        }

        model_train_status = ModelTrainStatus(**model_train_status_params)

        # Save checkpoint to a temporary file first
        temp_path = self.__save_checkpoint_temp(epoch, loss["validation"])

        try:
            # Save to MongoDB and get the ModelEntry ID
            model_id = self.model_repo.save(self.identifier, self.model_schema, model_train_status, temp_path)

            Config.log_debug_activate and self.utils_logger.info("Model saved to MongoDB with ID: %s", model_id)

            return model_id
        except Exception as e:
            self.utils_logger.error("Failed to save model to MongoDB: %s", str(e))
            raise
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    Config.log_debug_activate and self.utils_logger.info(
                        "Temporary file %s removed after MongoDB upload", temp_path)
            except Exception as e:
                self.utils_logger.warning("Failed to remove temporary file %s: %s", temp_path, str(e))


    def close_repo(self):
        self.model_repo.close()