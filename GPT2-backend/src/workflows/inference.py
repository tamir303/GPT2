import yaml
from torch import nn
from torch.optim import Optimizer

from src.etc.config import Config
from src.etc.logger import CustomLogger

class InferencePipeline:
    def __init__(self, model: nn.Module, optimizer: Optimizer):
        self.logger = CustomLogger(
            log_name = "InferencePipeline",
            log_level = Config.log_level,
            log_filename="inference.log"
        ).get_logger()

        self.logger.info("Inference pipeline initialized.")
        self.model = model
        self.optimizer = optimizer

        model_id = self.__get_model_id()
        self.save_load_handler = LoadSaveUtilsClass(self.model, self.optimizer, model_id)

    def run(self):
        pass


    def __get_model_id(self):
        # Load the model from the checkpoint
        try:
            with open("inference.yaml", "r") as f:
                inference_dict = yaml.safe_load(f)
                if inference_dict is not None and "model_id" in inference_dict:
                    model_id = inference_dict["model_id"]
                    self.logger.info(f"Model: {model_id}")
                    return model_id
                else:
                    print("No 'model_id' found in training.yaml or file is empty.")
        except FileNotFoundError:
            print("Error: 'training.yaml' not found.")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def __model_loader(self, model_id: str):

