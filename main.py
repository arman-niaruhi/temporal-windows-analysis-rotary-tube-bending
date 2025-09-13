# main.py
from src.pipeline.extractor import DataExtractor
from src.pipeline.transformer import DataTransformer
from src.pipeline.visualizer import DataVisulizer


def main():
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
    # Extract all tranformed sensor_data
    df_machine_and_movement, df_sensor, df_machine, df_movements = (
        transformer.get_process_data()
    )
    # Extract all tranformed geometry_data
    df_arc, df_lin1, df_lin2, linear_df, all_geometry_data = (
        transformer.get_geometry_data()
    )

    # -----------------------------
    # 3. Visualize
    # -----------------------------
    vizualiser = DataVisulizer()
    vizualiser.multi_sensor_experiment(
        dfs=[df_machine_and_movement, df_sensor, df_arc],
        experiment_id=123,
        df_names=["Machine & Movement", "Sensors", "Geometry"],
        x_axes=[
            "index",
            "index",
            "Angle[degree]ORDistance[mm]",
        ],
    )


if __name__ == "__main__":
    main()
