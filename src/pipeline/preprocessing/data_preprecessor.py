from src.pipeline.preprocessing.extractor import DataExtractor
from src.pipeline.preprocessing.transformer import DataTransformer
from src.pipeline.preprocessing.loader import DataLoader

from src.logging.log_utils import log_function


class DataPreprocessPipeline:
    @classmethod
    @log_function
    def run(cls):
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

        df_machine_and_movement, df_sensor, df_machine, df_movements = transformer.get_process_data()
        df_arc, df_lin1, df_lin2, linear_df, all_geometry_data = transformer.get_geometry_data()
        
        # Suppose you already have all your DataFrames
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
        }

        loader = DataLoader("data/tube_geometry.db")
        loader.save_to_sqlite(dataframes)