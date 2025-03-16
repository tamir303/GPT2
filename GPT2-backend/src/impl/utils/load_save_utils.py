import os
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
    def __init__(self, model: nn.Module, optimizer: Optimizer, storage: str = "local", identifier: str = None):
        # Initialize logger for utils
        self.utils_logger = CustomLogger(
            log_name='Utils',
            log_level=logging.DEBUG,
            log_dir='app_logs',
            log_filename='utils.log'
        ).get_logger()

        self.model = model
        self.optimizer = optimizer
        self.storage = storage
        self.identifier = identifier

        try:
            self.model_repo = ModelRepository()
        except Exception as e:
            self.utils_logger.error("Error initializing ModelRepository: %s", str(e))
            self.model_repo = None

    def __save_checkpoint_local(self, epoch: int, loss: float, filename: str = "checkpoint.pth") -> str:
        """
        Save a model checkpoint to a local file.

        Args:
            epoch (int): Current epoch.
            loss (float): Current loss.
            filename (str): Path to save checkpoint.

        Returns:
            str: Full path of the saved checkpoint file.
        """
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }

            checkpoint_dir = os.path.abspath(Config.checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            full_path = os.path.join(checkpoint_dir, filename)
            torch.save(checkpoint, full_path)

            Config.log_debug_activate and self.utils_logger.info("Checkpoint saved at %s for epoch %d, loss %.4f", full_path, epoch, loss)

            return full_path
        except Exception as e:
            self.utils_logger.error("Error saving checkpoint to %s: %s", full_path, str(e))
            raise

    def __load_checkpoint_local(self, filename: str = "checkpoint.pth") -> Tuple[int, float, str]:
        """
        Load a model checkpoint from a local file.

        Args:
            filename (str): Path to checkpoint file.

        Returns:
            Tuple[int, float]: (epoch, loss) from the checkpoint.
        """
        try:
            checkpoint_dir = os.path.abspath(Config.checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            full_path = os.path.join(checkpoint_dir, filename)
            checkpoint = torch.load(full_path)

            if checkpoint is None or not isinstance(checkpoint, dict):
                raise ValueError("Invalid checkpoint format")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch: int = checkpoint['epoch']
            loss: float = checkpoint['loss']
            self.utils_logger.info("Checkpoint loaded from %s. Resuming from epoch %d, loss %.4f", full_path, epoch, loss)
            return epoch, loss, filename

        except FileNotFoundError:
            self.utils_logger.warning("Checkpoint file %s not found, starting from scratch", full_path)
            return 0, float("inf"), filename

        except Exception as e:
            self.utils_logger.error("Error loading checkpoint from %s: %s", full_path, str(e))
            return 0, float("inf"), filename


    def load_checkpoint(self) -> Tuple[int, float, str]:
        """
        Load a model checkpoint from either local storage or MongoDB.

        Returns:
            Tuple[int, float]: (epoch, loss) from the checkpoint.

        Raises:
            ValueError: If storage type is invalid or MongoDB ID is not found.
        """
        if self.storage == "local":
            return self.__load_checkpoint_local()
        elif self.storage == "mongo":
            # Load from MongoDB
            try:
                model_schema, train_status_schema, checkpoint_io = self.model_repo.load(self.identifier)
                self.utils_logger.info(f"Loading model... \n schema: {model_schema}, train status: {train_status_schema}")
                checkpoint = torch.load(checkpoint_io)

                if checkpoint is None or not isinstance(checkpoint, dict):
                    raise ValueError("Invalid checkpoint format from MongoDB")
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']

                Config.log_debug_activate and self.utils_logger.info("Checkpoint loaded from MongoDB ID %s. Resuming from epoch %d, loss %.4f",
                                  self.identifier, epoch, loss)

                return epoch, loss, self.identifier
            except ValueError as ve:
                self.utils_logger.error("Error loading checkpoint from MongoDB ID %s: %s",self. identifier, str(ve))
                return 0, float("inf"), self.identifier
            except Exception as e:
                self.utils_logger.error("Unexpected error loading from MongoDB ID %s: %s", self.identifier, str(e))
                return 0, float("inf"), self.identifier
        else:
            raise ValueError(f"Invalid storage type: {self.storage}. Use 'local' or 'mongo'")


    def save_checkpoint(self, epoch: int, loss: dict, accuracy: float,) -> str:
        """
            Save a model checkpoint to either local storage or MongoDB.

            Args:
                epoch (int): Current epoch.
                loss (float): Current loss.
                accuracy (float): Current accuracy.

            Returns:
                str: Full path (local) or MongoDB ModelEntry ID.

            Raises:
                ValueError: If storage type is invalid or schema is missing for MongoDB.
            """
        if self.storage == "local":
            return self.__save_checkpoint_local(epoch, loss["validation"])
        elif self.storage == "mongo":
            # Create Pydantic ModelSchema
            model_schema_params = {
                "type": Config.name,
                "d_model": Config.d_model,
                "block_size": Config.block_size,
                "n_heads": Config.n_heads,
                "n_layers": Config.n_layers,
                "dropout": Config.dropout,
            }

            model_schema = ModelSchema(**model_schema_params)

            # Create ModelTrainStatus
            model_train_status_params = {
                "current_epoch": epoch,
                "val_loss": loss["validation"],
                "train_loss": loss["train"],
                "accuracy": accuracy
            }

            model_train_status = ModelTrainStatus(**model_train_status_params)

            # Save checkpoint locally first, then upload to MongoDB
            temp_path = self.__save_checkpoint_local(epoch, loss["validation"])

            # Save to MongoDB and get the ModelEntry ID
            model_id = self.model_repo.save(model_schema, model_train_status ,temp_path)

            # Clean up temporary file
            try:
                os.remove(temp_path)
                Config.log_debug_activate and self.utils_logger.info("Temporary file %s removed after MongoDB upload", temp_path)
            except Exception as e:
                self.utils_logger.warning("Failed to remove temporary file %s: %s", temp_path, str(e))

            return model_id
        else:
            raise ValueError(f"Invalid storage type: {self.storage}. Use 'local' or 'mongo'")
