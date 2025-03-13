import logging
import logging.handlers
import os
from datetime import datetime
from src.etc.config import Config

class CustomLogger:
    def __init__(self,
                 log_name='app_logger',
                 log_level=Config.log_level,
                 log_dir=Config.log_dir,
                 log_filename=None
        ):
        """
        Initialize the custom logger

        Args:
            log_name (str): Name of the logger
            log_level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir (str): Directory to store log files
            log_filename (str): Custom filename for log file (optional)
        """
        # Create logger
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)

        # Clear existing handlers to prevent duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Set default filename if none provided
        if log_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'application_{timestamp}.log'

        # Create file handler
        file_handler = logging.FileHandler(
            os.path.join(log_dir, log_filename),
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Set formatter for both handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Disable propagation to prevent double logging via root logger
        self.logger.propagate = False

    def get_logger(self):
        """Return the configured logger instance"""
        return self.logger


# Optional: Define HTTP handler (but attach it explicitly if needed)
handler = logging.handlers.HTTPHandler('log-server:9000', '/logs', method='POST')