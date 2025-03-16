import logging

import torch

from src.etc.logger import CustomLogger
from src.impl import init_tokenizer, init_model, init_trainer, get_loss, initialize_optimizer, split_data, get_data_loader
from src.etc.config import HyperParams, Config

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
        train, test = split_data(tensor_data, self.config.split_ratio)

        # Train the model using the training data
        self.trainer.train(train)

        # Evaluate the model and log test loss
        test_loss = get_loss(self.model, test)
        self.logger.info("Test Loss: %.4f", test_loss)