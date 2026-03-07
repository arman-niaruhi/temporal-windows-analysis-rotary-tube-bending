import logging
import json
import os

from src.logging.logging_config import setup_logging

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
    save_validation_confusion_matrices,
)


logger = logging.getLogger(__name__)


activity_recognition_config_path = (
    "config/machine-activity-recognition/machine-activity-recognition-config.json"
)


def main():
    setup_logging()
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
                        model_path_root=config.get("model_path_root"),
                        database_path=config.get("database_path"),
                        annotation_json_path=config.get("annotation_json_path"),
                        experiment_ids_path=config.get("experiment_ids_path"),
                        process_part=config.get("process_part"),
                        eliminated_columns=config.get("eliminated_columns"),
                        label=lbl,
                        pipeline_config=config.get("pipeline_config"),
                    )
                )
        else:
            model, sensors_df, test_loader, device, feature_cols = training_pipeline(
                model_path_root=config.get("model_path_root"),
                database_path=config.get("database_path"),
                annotation_json_path=config.get("annotation_json_path"),
                experiment_ids_path=config.get("experiment_ids_path"),
                process_part=config.get("process_part"),
                eliminated_columns=config.get("eliminated_columns"),
                label=config.get("label"),
                pipeline_config=config.get("pipeline_config"),
            )
    except Exception as e:
        logger.error(f"Error during training pipeline: {e}")
        return

    if config.get("analytics", False):
        try:
            analyze_features_result_path = config.get("analyze_features_result_path")
            analyze_features_result_path = os.path.join(
                analyze_features_result_path,
                config.get("process_part"),
                config.get("label"),
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
            config.get("show_result_properties"),
            config.get("process_part"),
            config.get("label"),
        )
    except Exception as e:
        logger.error(f"Warning: Plotting predictions failed: {e}")

    inference_config = config.get("inference_one_label_in_one", None)

    if inference_config:
        try:
            save_validation_confusion_matrices(
                database_path=config.get("database_path"),
                annotation_json_path=config.get("annotation_json_path"),
                experiment_ids_path=config.get("experiment_ids_path"),
                eliminated_columns=config.get("eliminated_columns"),
                models_path=inference_config.get("models_path"),
                model_config=config.get("pipeline_config").get("model_config"),
                labels=inference_config.get("labels"),
                process_part=config.get("process_part", "machine_and_movement"),
                save_dir_path=inference_config.get("save_dir_path"),
                batch_size=config.get("pipeline_config", {})
                .get("dataloader_config", {})
                .get("batch_size", 8),
            )

            TEST_EXPERIMENT_IDS = [
                2,
                3,
                22,
                23,
                40,
                54,
                83,
                85,
                110,
                112,
                119,
                120,
                121,
                122,
                123,
                178,
                179,
                182,
                183,
                211,
                212,
                213,
                255,
                258,
                261,
                271,
                272,
                273,
                302,
                303,
                304,
                317,
                318,
            ]
            for i in TEST_EXPERIMENT_IDS:
                inference_one_label_in_one(
                    exp_id=i,
                    database_path=config.get("database_path"),
                    annotation_json_path=config.get("annotation_json_path"),
                    eliminated_columns=config.get("eliminated_columns"),
                    models_path=inference_config.get("models_path"),
                    model_config=config.get("pipeline_config").get("model_config"),
                    labels=inference_config.get("labels"),
                    process_part=config.get("process_part", "machine_and_movement"),
                    save_dir_path=inference_config.get("save_dir_path"),
                    get_all_predictions_fn=get_all_predictions,
                    figsize=(15, 10),
                )
        except Exception as e:
            logger.error(
                f"Warning: Inference for one label in one experiment failed: {e}"
            )


if __name__ == "__main__":
    main()
