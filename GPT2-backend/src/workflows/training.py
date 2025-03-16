import logging
import yaml

from src.etc.logger import CustomLogger
from src.impl import init_tokenizer, init_model, init_trainer, get_loss, initialize_optimizer, split_data, get_data_loader
from src.etc.config import HyperParams, Config
from src.impl.utils.load_save_utils import LoadSaveUtilsClass


class TrainerPipeline:
    def __init__(self, file_path: str):
        # Set up a dedicated logger for the pipeline
        self.logger = CustomLogger(
            log_name='Tokenizer',
            log_level=logging.DEBUG,
            log_dir='app_logs',
            log_filename='tokenizer.log'
        ).get_logger()

        # Fetch configurations
        self.config: HyperParams = Config
        self.logger.info("Configurations fetched.")

        # Initialize Tokenizer, Model, Optimizer, and Trainer
        self.logger.info("Initializing Tokenizer...")
        self.tokenizer = init_tokenizer(file_path)

        self.logger.info("Initializing Model...")
        self.model = init_model(self.tokenizer.get_vocab_size())

        self.logger.info("Initializing Optimizer...")
        self.optimizer = initialize_optimizer(self.model, self.config)

        model_id = self.__get_model_id()
        self.save_load_handler = LoadSaveUtilsClass(self.model, self.optimizer, model_id)

        self.current_epoch, self.current_loss = 0, float("inf")
        if Config.load_existing_model:
            self.current_epoch, self.current_loss = self.__model_loader()

        self.logger.info("Initializing Trainer...")
        self.trainer = init_trainer(self.model, self.optimizer)

        self.logger.info("Loading text data using DataLoader...")
        self.data_loader = get_data_loader(file_path)

    def run(self):
        # Combine the loaded lines into a single text string
        data_text = "".join(self.data_loader.get_file_content())
        self.logger.debug("Loaded data text with length: %d", len(data_text))

        # Encode the text into a tensor representation
        tensor_data = self.tokenizer.encode(data_text)

        # Split data into training and testing sets
        train_data, test_data = split_data(tensor_data, self.config.split_ratio)

        # Train the model using the training data
        self.trainer.train(
            train_data,
            current_epoch = self.current_epoch,
            current_loss = self.current_loss,
            save_callable = self.save_load_handler.save_checkpoint
        )

        # Evaluate the model and log test loss
        test_loss = get_loss(self.model, test_data)
        self.logger.info("Test Loss: %.4f", test_loss)

    def __get_model_id(self):
        # Load the model from the checkpoint
        try:
            with open("training.yaml", "r") as f:
                training_dict = yaml.safe_load(f)
                if training_dict is not None and "model_id" in training_dict:
                    model_id = training_dict["model_id"]
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

    def __model_loader(self) -> tuple[int, float]:
        # Load the model from the checkpoint
        if Config.load_existing_model:  # Assuming Config is defined elsewhere
            try:
                with open("training.yaml", "r") as f:
                    training_dict = yaml.safe_load(f)
                    if training_dict is not None and "model_id" in training_dict:
                        epoch, loss, _= self.save_load_handler.load_checkpoint()
                        self.logger.info(f"Model and Optimizer loaded from checkpoint with model_id: {self.save_load_handler.identifier}")
                        return epoch, loss
                    else:
                        print("No 'model_id' found in training.yaml or file is empty.")
                        print("Training will start from scratch.")
                        return 0, float("-inf")
            except FileNotFoundError:
                print("Error: 'training.yaml' not found.")
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")