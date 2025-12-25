import argparse
import os
from src.logging.logging_config import setup_logging
setup_logging()
from src.pipeline.preprocessing.data_preprecessor import DataPreprocessPipeline
from src.pipeline.ml.classification.utils.plot_utils import (
    plot_predictions_vs_true_annot,
)
from src.pipeline.ml.classification.utils.training_utils import (
    analyze_features,
    training_pipeline,
)
from src.pipeline.ml.classification.utils.inference_one_label import (
    get_all_predictions,
    inference_one_label_in_one,
)
from src.pipeline.ml.context_extractor.utils.seed_utils import enforce_reproducibility
from src.pipeline.ml.context_extractor.utils.data.data_preprocessor import prepare_data
from src.pipeline.ml.context_extractor.utils.training_utils import train_model

import json

import logging

logger = logging.getLogger(__name__)


def main():
    activity_recognition_config_path = "config/machine_activity_recognition.json"
    parser = argparse.ArgumentParser(description="Run different pipeline steps.")
    parser.add_argument(
        "step",
        choices=["preprocess", "activity_recognition", "context_extraction"],
        help="Choose which pipeline step to run",
    )
    args = parser.parse_args()

    if args.step == "preprocess":
        with open("config/preprocessing_config.json", "r") as f:
            config = json.load(f)
        DataPreprocessPipeline.run(
            failed_experiment=config["failed_experiment"],
            eliminated_columns=config["eliminated_columns"],
            normalized_tables=config["normalized_tables"],
            nan_handler=True,
            correlation_matrices=config["correlation_matrices"],
        )

    elif args.step == "activity_recognition":
        logger.info("Starting activity recognition pipeline...")
        if not os.path.exists(activity_recognition_config_path):
            logger.error(
                f"Configuration file not found at {activity_recognition_config_path}. Please create the config file."
            )

        with open(activity_recognition_config_path, "r") as f:
            config = json.load(f)

        try:
            if config.get("label", "All_and_One") == "All_and_One":
                for lbl in [
                    "Clamping",
                    "De-Clamping",
                    "Mandrel Extraction",
                    "Bending",
                    "All",
                ]:
                    logger.info(f"Starting Train for individuall label: {lbl}")
                    model, sensors_df, test_loader, device, feature_cols = (
                        training_pipeline(
                            model_path_root=config.get(
                                "model_path_root", "models/classifier"
                            ),
                            database_path=config.get(
                                "database_path", "data/processed/tube_geometry.db"
                            ),
                            annotation_json_path=config.get(
                                "annotation_json_path",
                                "data/ml/machine-and-movement_complete.json",
                            ),
                            experiment_ids_path=config.get(
                                "experiment_ids_path",
                                "data/ml/unique_experiment_ids.json",
                            ),
                            machine_part=config.get(
                                "machine_part", "machine_and_movement"
                            ),
                            eliminated_columns=config.get(
                                "eliminated_columns",
                                [
                                    "PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]",
                                    "COLLET_ROTATING_Movement_[mm]",
                                    "BEND-DIE_VERTICAL_Movement_[mm]",
                                    "PRESSURE-DIE_LATERAL_Movement_[mm]",
                                ],
                            ),
                            label=lbl,
                            pipeline_config=config.get(
                                "pipeline_config",
                                {
                                    "dataloader_config": {"batch_size": 8},
                                    "model_config": {
                                        "hidden_size": 64,
                                        "num_layers": 2,
                                    },
                                    "training_config": {
                                        "training": False,
                                        "num_epochs": 1,
                                        "learning_rate": 1e-5,
                                        "patience": 3,
                                    },
                                },
                            ),
                        )
                    )
            else:
                model, sensors_df, test_loader, device, feature_cols = (
                    training_pipeline(
                        model_path_root=config.get(
                            "model_path_root", "models/classifier"
                        ),
                        database_path=config.get(
                            "database_path", "data/processed/tube_geometry.db"
                        ),
                        annotation_json_path=config.get(
                            "annotation_json_path",
                            "data/ml/machine-and-movement_complete.json",
                        ),
                        experiment_ids_path=config.get(
                            "experiment_ids_path", "data/ml/unique_experiment_ids.json"
                        ),
                        machine_part=config.get("machine_part", "machine_and_movement"),
                        eliminated_columns=config.get(
                            "eliminated_columns",
                            [
                                "PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]",
                                "COLLET_ROTATING_Movement_[mm]",
                                "BEND-DIE_VERTICAL_Movement_[mm]",
                                "PRESSURE-DIE_LATERAL_Movement_[mm]",
                            ],
                        ),
                        label=config.get("label", "All"),
                        pipeline_config=config.get(
                            "pipeline_config",
                            {
                                "dataloader_config": {"batch_size": 8},
                                "model_config": {"hidden_size": 64, "num_layers": 2},
                                "training_config": {
                                    "training": False,
                                    "num_epochs": 1,
                                    "learning_rate": 1e-5,
                                    "patience": 3,
                                },
                            },
                        ),
                    )
                )
        except Exception as e:
            logger.error(f"Error during training pipeline: {e}")
            return

        if config.get("analytics", False):
            try:
                analyze_features_result_path = config.get(
                    "analyze_features_result_path", "results/analyze_features"
                )
                analyze_features_result_path = os.path.join(
                    analyze_features_result_path,
                    config.get("machine_part", "machine_and_movement"),
                    config.get("label", "All"),
                )
                os.makedirs(analyze_features_result_path, exist_ok=True)
                analyze_features(
                    analyze_features_result_path, model, sensors_df, test_loader, device
                )
            except Exception as e:
                logger.error(f"Warning: Feature analysis failed: {e}")

        try:
            plot_predictions_vs_true_annot(
                model,
                getattr(test_loader, "dataset", None),
                sensors_df,
                feature_cols,
                config.get(
                    "show_result_properties",
                    {
                        "store_plots": True,
                        "store_plots_path": "results/activity_recognition",
                    },
                ),
                config.get("machine_part", "machine_and_movement"),
                config.get("label", "All"),
            )
        except Exception as e:
            logger.error(f"Warning: Plotting predictions failed: {e}")

        inference_config = config.get("inference_one_label_in_one", None)

        if inference_config:
            try:
                inference_one_label_in_one(
                    exp_id=110,
                    database_path=config.get("database_path", "data/processed/tube_geometry.db"),
                    annotation_json_path=config.get(
                        "annotation_json_path",
                        "data/ml/machine-and-movement_complete.json",
                    ),
                    eliminated_columns=config.get(
                        "eliminated_columns",
                        [
                            "PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]",
                            "COLLET_ROTATING_Movement_[mm]",
                            "BEND-DIE_VERTICAL_Movement_[mm]",
                            "PRESSURE-DIE_LATERAL_Movement_[mm]",
                        ],
                    ),
                    models_path=inference_config.get("models_path"),
                    model_config=config.get("pipeline_config").get("model_config",{"hidden_size": 64, "num_layers": 2}),
                    labels=inference_config.get("labels"),
                    machine_part=config.get("machine_part", "machine_and_movement"),
                    save_dir_path=inference_config.get("save_dir_path"),
                    get_all_predictions_fn=get_all_predictions,
                    figsize=(15, 10),
                )
            except Exception as e:
                logger.error(
                    f"Warning: Inference for one label in one experiment failed: {e}"
                )

    elif args.step == "context_extraction":
        extraction_configuration_path = "config/context_extraction_config.json"
        with open(extraction_configuration_path, "r") as f:
            config = json.load(f)
        enforce_reproducibility(seed=config.get("general_setting").get("seed"))
        input_path_param = config.get("input_path_param")
        preprocessing_param = config.get("preprocessing_param")
        machine_part = input_path_param.get("machine_part")

        (
            X,
            Y,
            sensor_names,
            target_feature_names,
            annot_timesteps,
            mandrel_extraction_annot_timesteps,
        ) = prepare_data(
            input_path_param=input_path_param, preprocessing_param=preprocessing_param
        )

        train_model(
            X=X,
            Y=Y,
            params=config.get("training_param"),
            sensor_names=sensor_names,
            target_feature_names=target_feature_names,
            machine_part=machine_part,
            preprocessing_info=config.get("preprocessing_param"),
            annot_timesteps=annot_timesteps,
            mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
        )


if __name__ == "__main__":
    main()
    # python main.py preprocess
    # python main.py activity_recognition
    # python main.py context_extraction
