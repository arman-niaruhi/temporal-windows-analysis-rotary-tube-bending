from src.pipeline.preprocessing.extractor import DataExtractor
from src.pipeline.preprocessing.transformer import DataTransformer
from src.pipeline.preprocessing.loader import DataLoader

from src.logging.log_utils import log_function

class DataPreprocessPipeline:
    """
    Pipeline for extracting, transforming, and loading bending setup and process data.

    This class orchestrates the full ETL process:
    1. Extracts bending setup and machine/process data using DataExtractor.
    2. Transforms the data with DataTransformer (quality check, remove failed experiments, normalization, NaN handling).
    3. Loads the final DataFrames into a SQLite database using DataLoader.
    """

    @classmethod
    @log_function
    def run(cls):
        """
        Execute the full preprocessing pipeline: extraction, transformation, and loading.

        Steps:
        ------
        1. Extraction:
            - Use DataExtractor to load all bending setups into DataFrames.
        2. Transformation:
            - Create a DataTransformer instance with the extracted DataFrames.
            - Check data quality and convert types.
            - Remove failed experiments (hardcoded IDs [1, 48, 166]).
            - Normalize numeric data.
            - Drop all-NaN columns.
            - Retrieve processed process and geometry DataFrames.
            - Eliminate columns that are always constant(PRESSURE-DIE_LEFT_AXIAL_Movement_[mm], COLLET_ROTATING_Movement_[mm])
        3. Loading:
            - Collect selected DataFrames into a dictionary.
            - Use DataLoader to save them to SQLite.
            - Certain tables store the index as a column for query convenience.

        Args:
            None

        Returns:
            None: The pipeline updates and saves the DataFrames to the SQLite database.
        """
        # -----------------------------
        # 1. Extraction
        # -----------------------------
        extractor = DataExtractor()
        dfs = extractor.get_all_bending_setups()

        # -----------------------------
        # 2. Transformation
        # -----------------------------
        transformer = DataTransformer(**dfs)
        transformer.check_quality()
        transformer.delete_failed_experiment(failed_experiment=[1, 48, 166])
        transformer.normalize_data()
        transformer.nan_handler()
        transformer.save_correlation_matrices(tables=["df_machine", "df_sensor","df_movements"])
        transformer.eliminate_column(df_name="df_movements", column_name="PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]")
        transformer.eliminate_column(df_name="df_movements", column_name="COLLET_ROTATING_Movement_[mm]")
        
        df_machine_and_movement, df_sensor, df_machine, df_movements = transformer.get_process_data()
        df_arc, df_lin1, df_lin2, linear_df, all_geometry_data = transformer.get_geometry_data()
        df_bending = transformer.get_bending_setup()
        
        # -----------------------------
        # 3. Loading
        # -----------------------------
        dataframes = {
            "df_machine_and_movement": df_machine_and_movement,
            #"df_lin1": df_lin1,
            #"df_lin2": df_lin2,
            "df_arc": df_arc,
            #"all_geometry_data": all_geometry_data,
            #"linear_df": linear_df,
            "df_machine": df_machine,
            "df_sensor": df_sensor,
            "df_movements": df_movements,
            "df_bending": df_bending
        }

        loader = DataLoader("data/tube_geometry.db")
        loader.store_to_sqlite(
            dataframes,
            store_index_tables=["df_machine", "df_sensor", "df_machine_and_movement", "df_movements"]
        )
