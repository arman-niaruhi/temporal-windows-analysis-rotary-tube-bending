from src.pipeline.preprocessing.data_preprecessor import DataPreprocessPipeline
from src.pipeline.preprocessing.loader import DataLoader
from src.pipeline.report.visualizer import DataVisulizer
from src.pipeline.ml.context_extactor import ContextExtractor
import streamlit as st
import json


def preprocess_data(
    failed_experiment, eliminated_columns, normalized_tables, correlation_matrices
):
    DataPreprocessPipeline.run(
        failed_experiment=failed_experiment,
        eliminated_columns=eliminated_columns,
        normalized_tables=normalized_tables,
        nan_handler=True,
        correlation_matrices=correlation_matrices,
    )


@st.cache_data
def load_data():
    loader = DataLoader("data/processed/tube_geometry.db")
    return loader.load_all_data_from_sqlite()


def export_csv(loaded_dfs):
    df_arc = loaded_dfs["arc"]
    df_machine_and_movement = loaded_dfs["machine_and_movement"]
    load_setup = loaded_dfs.get("bending", None)

    cols_to_match = [
        "Pressure-die lateral position",
        "Pressure-die distance",
        "Pressure-die boost",
        "Mandrel position",
        "Mandrel retraction timing",
        "Collet boost",
        "Clamp-die lateral position",
    ]

    loader = DataLoader("data/processed/tube_geometry.db")
    loader.store_to_csv(
        cols_to_match=cols_to_match,
        load_setup=load_setup,
        selected_dfs_features=[df_machine_and_movement],
        selected_dfs_target=[df_arc],
        feature_file="data/ml/features.csv",
        target_file="data/ml/targets.csv",
    )


def extract_context():
    # Implement your context extraction here
    pass


def visualize_data():
    vizualiser = DataVisulizer()
    vizualiser.interactive_plot_streamlit()


def main():
    # -----------------------------
    # 1. Preprocessing
    # -----------------------------
    # Load the configuration
    # with open("config/config.json", "r") as f:
    #     config = json.load(f)

    # eliminated_columns = config["eliminated_columns"]
    # failed_experiment = config["failed_experiment"]
    # normalized_tables = config["normalized_tables"]
    # correlation_matrices = config["correlation_matrices"]
    # preprocess_data(failed_experiment, eliminated_columns, normalized_tables, correlation_matrices)

    # -----------------------------
    # 2. CSV Export
    # -----------------------------
    # loaded_dfs = load_data()
    # export_csv(loaded_dfs)

    # -----------------------------
    # 3. Context Extraction
    # -----------------------------
    # extract_context()

    # -----------------------------
    # 4. Visualization
    # -----------------------------
    visualize_data()


if __name__ == "__main__":
    main()
    # streamlit run main.py
