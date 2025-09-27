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
    load_setup = loaded_dfs.get("df_bending", None)
    import pandas as pd
    from collections import defaultdict

    # Columns to match
    cols_to_match = [
        'Pressure-die lateral position', 
        'Pressure-die distance', 
        'Pressure-die boost', 
        'Mandrel position', 
        'Mandrel retraction timing', 
        'Collet boost', 
        'Clamp-die lateral position'
    ]

    # Group by these columns and collect Experiment_IDs
    grouped = load_setup.groupby(cols_to_match)['Experiment_ID'].apply(list)

    # Filter only groups with more than 1 experiment (i.e., duplicates)
    duplicates_list = [exp_list for exp_list in grouped if len(exp_list) > 1]

    # Show the result
    print(duplicates_list)
    
    # Take only the first experiment ID from each group
    first_experiments = [exp_list[0] for exp_list in grouped]

    # Filter the machine and movement DataFrame
    df_machine_and_movement = loaded_dfs["df_machine_and_movement"]
    df_machine_filtered = df_machine_and_movement[df_machine_and_movement['Experiment_ID'].isin(first_experiments)]
    df_arc_filtered = df_arc[df_arc['Experiment_ID'].isin(first_experiments)]
    df_machine_filtered.to_csv("features.csv")
    df_arc_filtered.to_csv("targets.csv")
    
    # -----------------------------
    # 2. Context Extraction
    # -----------------------------

    # -----------------------------
    # 3. Visualize
    # -----------------------------
    vizualiser = DataVisulizer()
    vizualiser.interactive_plot_streamlit()


if __name__ == "__main__":
    main()
    # streamlit run main.py 
