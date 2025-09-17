from src.pipeline.preprocessing.data_preprecessor import DataPreprocessPipeline
from src.pipeline.preprocessing.loader import DataLoader
from src.pipeline.report.visualizer import DataVisulizer
from src.pipeline.ml.context_extactor import ContextExtractor


def main():
    # -----------------------------
    # 3. Load Data From SQLite
    # -----------------------------
    #DataPreprocessPipeline.run()
    EXPERIMENT_ID = 8
    # loader = DataLoader("data/tube_geometry.db")
    # loaded_dfs = loader.load_data_by_experiment(EXPERIMENT_ID)
    # df_arc = loaded_dfs["df_arc"]
    # df_machine_and_movement = loaded_dfs["df_machine_and_movement"]

    # -----------------------------
    # 2. Context Extraction
    # -----------------------------
    #context_extractor = ContextExtractor(input_df=df_sensor_2, target_df=df_arc_2)
    #context_extractor.extract_important_window(target_column="Collapse [mm]", num_top_windows=5, )

    # -----------------------------
    # 3. Visualize
    # -----------------------------
    vizualiser = DataVisulizer()
    # vizualiser.multi_sensor_experiment(
    #     dfs=[df_machine_and_movement, df_arc],
    #     experiment_id=EXPERIMENT_ID,
    #     df_names=["Machine & Movement", "Geometry"],
    #     x_axes=[
    #         "index",
    #         "Angle[degree]ORDistance[mm]",
    #     ],
    # )

    vizualiser.interactive_plot_streamlit()


if __name__ == "__main__":
    main()
    
    # streamlit run main.py 
