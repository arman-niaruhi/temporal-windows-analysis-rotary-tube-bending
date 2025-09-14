from src.pipeline.preprocessing.extractor import DataExtractor
from src.pipeline.preprocessing.transformer import DataTransformer
from src.pipeline.report.visualizer import DataVisulizer
from src.pipeline.ml.context_extactor import ContextExtractor


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
    transformer.normalize_data()
    
    df_machine_and_movement, df_sensor, _, _ = transformer.get_process_data(experiment_ids=[60])
    df_arc, _, _, _, _ = transformer.get_geometry_data(experiment_ids=[60])

    # -----------------------------
    # 3. Context Extraction
    # -----------------------------
    # context_extractor = ContextExtractor(input_df=df_machine_and_movement, target_df=df_arc)
    # context_extractor.extract_important_window(target_column="Collapse [mm]")

    # -----------------------------
    # 4. Visualize
    # -----------------------------
    vizualiser = DataVisulizer()
    # vizualiser.multi_sensor_experiment(
    #     dfs=[df_machine_and_movement, df_arc],
    #     experiment_id=60,
    #     df_names=["Machine & Movement", "Geometry"],
    #     x_axes=[
    #         "index",
    #         "Angle[degree]ORDistance[mm]",
    #     ],
    # )
    vizualiser.interactive_plot(
        dfs=[df_machine_and_movement, df_arc],
        df_names=["Machine & Movement", "Geometry"],
        x_axes=[
            "index",
            "Angle[degree]ORDistance[mm]",
        ],
    )


if __name__ == "__main__":
    main()
