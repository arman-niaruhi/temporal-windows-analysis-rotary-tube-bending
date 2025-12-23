import argparse
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


def main():
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
        with open("config/machine_activity_recognition.json", "r") as f:
            config = json.load(f)
        model, sensors_df, test_loader, device, feature_cols = training_pipeline(
            model_path_root=config["model_path_root"],
            database_path=config["database_path"],
            annotation_json_path=config["annotation_json_path"],
            experiment_ids_path=config["experiment_ids_path"],
            machine_part=config["machine_part"],
            eliminated_columns=config["eliminated_columns"],
            label=config["label"],
            pipeline_config=config["pipeline_config"],
        )

        if config["analytics"]:
            analyze_features(model, sensors_df, test_loader, device)

        plot_predictions_vs_true_annot(
            model,
            test_loader.dataset,
            sensors_df,
            feature_cols,
            config["show_result_properties"],
            config["machine_part"],
            config["label"],
        )
        if config["inference_one_label_in_one"]:
            inference_one_label_in_one(
                110,
                config["database_path"],
                config["annotation_json_path"],
                config["eliminated_columns"],
                config["inference_one_label_in_one"].get("models_path"),
                config["inference_one_label_in_one"].get("labels"),
                config["machine_part"],
                get_all_predictions,
                figsize=(15, 10),
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
            X,
            Y,
            config.get("training_param"),
            sensor_names,
            target_feature_names,
            machine_part,
            config.get("preprocessing_param"),
            annot_timesteps,
            mandrel_extraction_annot_timesteps,
        )


if __name__ == "__main__":
    main()
    # python main.py preprocess
    # python main.py activity_recognition
    # python main.py context_extraction
