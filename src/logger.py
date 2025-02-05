import logging


def get_logger() -> logging.Logger:
    # Configure a logger for the project
    logger = logging.getLogger("mlops_project")
    logger.setLevel(logging.DEBUG)

    # You can add handlers to log to file or stdout as needed
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
