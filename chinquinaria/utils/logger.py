import logging
from chinquinaria.config import CONFIG

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if CONFIG.get("debug", False):
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    return logger