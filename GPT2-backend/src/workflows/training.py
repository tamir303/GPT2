import logging
from src.etc.logger import CustomLogger
from src.impl import init_tokenizer, init_model, init_trainer, get_loss, initialize_optimizer, split_data
from src.etc.config import HyperParams, Config
from data import load_text_file

class TrainerPipeline:
    def __init__(self, file_path: str):
        # Initialize logger at module level
        self.logger = CustomLogger(
            log_name='Tokenizer',
            log_level=logging.DEBUG,
            log_dir='app_logs',
            log_filename='tokenizer.log'
        ).get_logger()

        logging.info("Fetching configurations...")
        self.config: HyperParams = Config

        logging.info("Initializing Tokenizer...")
        self.tokenizer = init_tokenizer(file_path)

        logging.info("Initializing Model...")
        self.model = init_model(self.tokenizer.get_vocab_size(), self.config)

        logging.info("Initializing Optimizer...")
        self.optimizer = initialize_optimizer(self.model, self.config)

        logging.info("Initializing Trainer...")
        self.trainer = init_trainer(self.model, self.optimizer)

    def run(self):
        data = load_text_file()
        tensor_data = self.tokenizer.encode(data)
        train, test = split_data(tensor_data, self.config.split_ratio)
        self.trainer.train(train)
        self.logger.info(f"Test Loss: {get_loss(self.model, test)}")