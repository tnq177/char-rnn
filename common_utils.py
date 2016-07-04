import logging

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    
    if not logger.handlers:
        debug_handler = logging.FileHandler('./DEBUG.log')
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)

    return logger
