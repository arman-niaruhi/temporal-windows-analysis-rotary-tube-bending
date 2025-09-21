from src.pipeline.preprocessing.data_preprecessor import DataPreprocessPipeline
from src.pipeline.preprocessing.loader import DataLoader
from src.pipeline.report.visualizer import DataVisulizer
from src.pipeline.ml.context_extactor import ContextExtractor
import streamlit as st


def main():
    # -----------------------------
    # 1. Load Data From SQLite or update the SQLite
    # -----------------------------
    # DataPreprocessPipeline.run()
    @st.cache_data
    def load_data():
        loader = DataLoader("data/tube_geometry.db")
        return loader.load_all_data()

    loaded_dfs = load_data()
    df_arc = loaded_dfs["df_arc"]
    df_machine_and_movement = loaded_dfs["df_machine_and_movement"]

    # -----------------------------
    # 2. Context Extraction
    # -----------------------------
    # context_extractor = ContextExtractor(input_df=df_machine_and_movement, target_df=df_arc)
    # context_extractor.extract_important_window(target_column="Collapse [mm]", num_top_windows=50, )

    # -----------------------------
    # 3. Visualize
    # -----------------------------
    vizualiser = DataVisulizer()
    vizualiser.interactive_plot_streamlit()


if __name__ == "__main__":
    main()
    # streamlit run main.py 
