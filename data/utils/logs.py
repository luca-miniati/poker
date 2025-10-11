import os
import logging
from datetime import datetime


def setup_logging(log_dir: str, script_name: str, level: str = logging.INFO) -> logging.Logger:
    '''
    Set up logging to both console and file.
    Returns the logger instance.
    '''
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    log_filename = os.path.join(
        log_dir,
        f'{script_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
    )
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f'Logging started. Log file: {log_filename}')
    return logger