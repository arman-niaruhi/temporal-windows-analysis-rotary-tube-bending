import logging

from src.logging.logging_config import setup_logging
from src.pipeline.preprocessing.data_preprecessor import DataPreprocessPipeline


logger = logging.getLogger(__name__)


# ========================================
# Configuration
# ========================================
# Experiment IDs removed from the ETL output because they are known failed runs.
# Value: list[int].
FAILED_EXPERIMENTS = [1, 48, 166]

# Columns removed during ETL because they are not useful for downstream analysis.
# Value: dict[str, list[str]] mapping source table names to column names.
ELIMINATED_COLUMNS = {
    "df_machine": [
        "MACHINE_PRESSURE-DIE_LEFT_AXIAL_Max_Torque_[%]",
        "MACHINE_COLLET_ROTATING_Max_Torque_[%]",
    ]
}

# Tables whose numeric values should be normalized during preprocessing.
# Value: list[str] of table names produced by the extractor/transformer.
NORMALIZED_TABLES = [
    "df_lin1",
    "df_lin2",
    "df_stl_arc",
    "df_stl_lin1",
    "df_stl_lin2",
    "df_machine",
    "df_sensor",
    "df_movement",
    "df_bending",
]

# Tables for which correlation matrices should be generated and saved.
# Value: list[str] of table names.
CORRELATION_MATRICES = [
    "df_machine",
    "df_movement",
    "df_arc",
]

# Whether NaN cleanup should run after transformation.
# Values: True or False.
ENABLE_NAN_HANDLER = True


def main():
    setup_logging()

    DataPreprocessPipeline.run(
        failed_experiment=FAILED_EXPERIMENTS,
        eliminated_columns=ELIMINATED_COLUMNS,
        normalized_tables=NORMALIZED_TABLES,
        nan_handler=ENABLE_NAN_HANDLER,
        correlation_matrices=CORRELATION_MATRICES,
    )


if __name__ == "__main__":
    main()
