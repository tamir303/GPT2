import yaml

from src.etc.config import Config, HyperParams
from src.etc.logger import CustomLogger
from src.impl import init_model, init_tokenizer, initialize_optimizer
from src.impl.utils.load_save_utils import LoadSaveUtilsClass


class InferencePipeline:
    def __init__(self, file_path: str):
        self.logger = CustomLogger(
            log_name = "InferencePipeline",
            log_level = Config.log_level,
            log_filename="inference.log"
        ).get_logger()

        self.logger.info("Initializing Inference pipeline...")
        self.config: HyperParams = Config

        self.logger.info("Initializing Tokenizer...")
        self.tokenizer = init_tokenizer(file_path)

        self.model = init_model(self.tokenizer.get_vocab_size())
        self.optimizer = initialize_optimizer(self.model, self.config)

        model_id = self.__get_model_id()
        self.save_load_handler = LoadSaveUtilsClass(self.model, self.optimizer, model_id)
        self.save_load_handler.load_checkpoint()

        self.logger.info("Inference pipeline ready.")

    def run(self, context: str):
        encoded_context = self.tokenizer.encode(context)
        generated_ids = self.model.generate(encoded_context, Config.max_tokens)
        generated_tokens = self.tokenizer.decode(generated_ids[0])
        return generated_tokens


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

