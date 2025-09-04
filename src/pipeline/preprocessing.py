from pathlib import Path
from src.logging.log_utils import log_function, logger

import pandas as pd

@log_function
def data_extractor(csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        logger.error(f"Path does not exist: {csv_path}")
        return
    if path.suffix.lower() != ".csv":
        logger.warning(f"Not a CSV file: {csv_path}")
        return
    logger.info(f"CSV file loaded from: {csv_path}")
    try:
        raw_df = pd.read_csv(csv_path)
        return raw_df
    except Exception as e:
        logger.error(f"Failed to read CSV: {csv_path} | Error: {e}")
        return pd.DataFrame()
    