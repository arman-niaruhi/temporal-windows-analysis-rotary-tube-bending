import logging
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.INFO,  # Must be INFO or lower
    format="%(asctime)s | %(levelname)s | %(funcName)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def log_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Finishing {func.__name__}")
        return result
    return wrapper