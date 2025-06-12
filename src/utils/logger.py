import logging
import os

os.makedirs("logs", exist_ok=True)

def setup_logger(name, log_file):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

error_logger = setup_logger("error", "logs/errors.log")
train_logger = setup_logger("train", "logs/training.log")
prediction_logger = setup_logger("predict", "logs/predictions.log")