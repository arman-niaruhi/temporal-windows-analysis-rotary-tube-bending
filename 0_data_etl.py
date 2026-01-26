import json

from src.logging.logging_config import setup_logging
from src.pipeline.preprocessing.data_preprecessor import DataPreprocessPipeline

import logging

logger = logging.getLogger(__name__)

     
def main():
    # ============================================================
    # Load schema and config and validate the config json entries and extract the required parameters
    # ============================================================
    setup_logging()
   
    with open("config/preprocessing/preprocessing_config.json", "r") as f:
        config = json.load(f)
    DataPreprocessPipeline.run(
        failed_experiment=config["failed_experiment"],
        eliminated_columns=config["eliminated_columns"],
        normalized_tables=config["normalized_tables"],
        nan_handler=True,
        correlation_matrices=config["correlation_matrices"],
    )

if __name__ == "__main__":
    main()
