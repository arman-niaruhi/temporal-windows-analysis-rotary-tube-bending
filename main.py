import argparse
from src.pipeline.preprocessing.data_preprecessor import DataPreprocessPipeline
from src.pipeline.preprocessing.loader import DataLoader
from src.pipeline.report.visualizer import DataVisulizer
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
    df_linear = loaded_dfs["linear"]
    df_linear1 = loaded_dfs["lin1"]
    df_linear2 = loaded_dfs["lin2"]
    df_machine_and_movement = loaded_dfs["machine_and_movement"]
    df_sensor = loaded_dfs["sensor"]
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
        selected_dfs_target=[df_linear1],
        feature_file="data/ml/features.csv",
        target_file="data/ml/targets.csv",
    )


def extract_context():
    # Placeholder for context extraction
    pass


def visualize_data():
    vizualiser = DataVisulizer()
    vizualiser.interactive_plot_streamlit()


def main():
    parser = argparse.ArgumentParser(description="Run different pipeline steps.")
    parser.add_argument(
        "step",
        choices=["preprocess", "export", "context", "visualize"],
        help="Choose which pipeline step to run",
    )
    args = parser.parse_args()

    if args.step == "preprocess":
        with open("config/config.json", "r") as f:
            config = json.load(f)
        preprocess_data(
            config["failed_experiment"],
            config["eliminated_columns"],
            config["normalized_tables"],
            config["correlation_matrices"],
        )

    elif args.step == "export":
        loaded_dfs = load_data()
        export_csv(loaded_dfs)

    elif args.step == "context":
        extract_context()

    elif args.step == "visualize":
        visualize_data()


if __name__ == "__main__":
    main()
    # Example usage:
    # python main.py preprocess
    # python main.py export
    # python main.py context
    # streamlit run main.py visualize
