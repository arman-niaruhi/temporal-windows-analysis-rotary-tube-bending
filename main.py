import argparse
from src.pipeline.preprocessing.data_preprecessor import DataPreprocessPipeline
from src.pipeline.ml.classification.utils.plot_utils import plot_predictions_vs_true_annot
from src.pipeline.ml.classification.training import analyze_features, training_pipeline 
from src.pipeline.ml.classification.inference_one_label import get_all_predictions, plot_experiment
from src.pipeline.ml.context_extractor.utils.lstm_utils.config.seed import enforce_reproducibility
from src.pipeline.ml.context_extractor.utils.lstm_utils.data.data_preprocessor import prepare_data
from src.pipeline.ml.context_extractor.utils.lstm_utils.models.training import train_model
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


def extract_context(extraction_configuration_path:str):
    with open(extraction_configuration_path, "r") as f:
        config = json.load(f)
    enforce_reproducibility(seed=config.get("general_setting").get("seed"))
    input_path_param = config.get("input_path_param")
    preprocessing_param = config.get("preprocessing_param")
    machine_part = input_path_param.get("machine_part")

    X, Y, sensor_names, target_feature_names, annot_timesteps, mandrel_extraction_annot_timesteps = prepare_data(
        input_path_param=input_path_param, preprocessing_param=preprocessing_param
    )
    
    train_model(
        X,
        Y,
        config.get("training_param"),
        sensor_names,
        target_feature_names,
        machine_part,
        config.get("preprocessing_param"),
        annot_timesteps, 
        mandrel_extraction_annot_timesteps
    )


def main():
    parser = argparse.ArgumentParser(description="Run different pipeline steps.")
    parser.add_argument(
        "step",
        choices=["preprocess", "activity_recognition_one", "activity_recognition", "context", "visualize"],
        help="Choose which pipeline step to run",
    )
    args = parser.parse_args()

    if args.step == "preprocess":
        with open("config/preprocessing_config.json", "r") as f:
            config = json.load(f)
        preprocess_data(
            config["failed_experiment"],
            config["eliminated_columns"],
            config["normalized_tables"],
            config["correlation_matrices"],
        )
        
    elif args.step == "activity_recognition":
        with open("config/machine_activity_recognition.json", "r") as f:
            config = json.load(f)
        model, sensors_df, test_loader, device, feature_cols= training_pipeline(model_path_root=config["model_path_root"],
        database_path=config["database_path"],
        annotation_json_path=config["annotation_json_path"],
        experiment_ids_path=config["experiment_ids_path"],
        machine_part=config["machine_part"],
        eliminated_columns=config["eliminated_columns"],
        label=config["label"],
        pipeline_config=config["pipeline_config"])

        # analyze_features(model, sensors_df, test_loader, device)

        EXPERIMENT_IDS = [
            # 2,
            # 3,
            # 22,
            # 23,
            # 40,
            # 54,
            # 83,
            # 85,
            # 110,
            # 112,
            # 119,
            # 120,
            # 121,
            # 122,
            # 123,
            # 178,
            # 179,
            # 182,
            # 183,
            # 211,
            # 212,
            # 213,
            # 255,
            # 258,
            # 261,
            # 271,
            # 272,
            # 273,
            # 302,
            # 303,
            # 304,
            # 317,
            # 318,
            110
        ]
        plot_predictions_vs_true_annot(model, test_loader.dataset, sensors_df, feature_cols, EXPERIMENT_IDS)
        
    elif args.step == "activity_recognition_one":
        import sys
        import os
        current_dir = os.getcwd()
        project_root = os.path.join(current_dir)
        project_root = os.path.abspath(project_root)
        sys.path.insert(0, project_root)
        

        DATABASE_PATH = f"{project_root}/data/processed/tube_geometry.db"
        MACHINE_PART = "machine_and_movement"
        ANNOTATION_JSON_PATH = f"{project_root}/data/ml/machine-and-movement_complete.json"
        ELIMINATED_COLUMNS = [
                                "PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]", 
                                "COLLET_ROTATING_Movement_[mm]", 
                                "BEND-DIE_VERTICAL_Movement_[mm]", 
                                "PRESSURE-DIE_LATERAL_Movement_[mm]"
                                ]

        LABELS = ["Clamping", "Bending", "Mandrel Extraction", "De-Clamping"]
        plot_experiment(
            110,
            DATABASE_PATH, 
            ANNOTATION_JSON_PATH,
            ELIMINATED_COLUMNS, 
            project_root, 
            LABELS,
            MACHINE_PART,
            get_all_predictions,
            figsize=(15, 10)
        )


    elif args.step == "context":
        extract_context("config/context_extraction_config.json")


if __name__ == "__main__":
    main()
    # python main.py preprocess
    # python main.py activity_recognition
    # python main.py activity_recognition_one
    # python main.py context
