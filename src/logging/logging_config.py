import logging
import sys


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",     
        logging.INFO: "\033[32m",      
        logging.WARNING: "\033[33m",   
        logging.ERROR: "\033[31m",     
        logging.CRITICAL: "\033[41m",  
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def setup_logging(level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        return  

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
