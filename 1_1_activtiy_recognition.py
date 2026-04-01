import logging
from pathlib import Path

from src.logging.logging_config import setup_logging

from src.pipeline.ml.classification.utils.plot_utils import (
    plot_predictions_vs_true_annot,
)
from src.pipeline.ml.classification.utils.training_utils import (
    training_pipeline,
)
from src.pipeline.ml.classification.utils.inference_one_label import (
    get_all_predictions,
    plot_all_and_one_as_multiclass,
    run_default_inference_plots,
    save_validation_confusion_matrices,
)


logger = logging.getLogger(__name__)


# ========================================
# Paths
# ========================================
BASE_PATH = Path("data")

# Processed sensor tables used for training and inference.
PROCESSED_DATA_DIRECTORY =  BASE_PATH / "processed"

# Annotation file exported from the annotation tool.
ANNOTATION_FILE_PATH = BASE_PATH / "ml" / "machine-and-movement_complete.json"

# Root directory for analytics, plots, confusion matrices, and inference figures.
RESULTS_DIRECTORY = Path("results") / "activity_recognition"

# ========================================
# Pipeline Selection
# ========================================
# Target label to train/evaluate.
# Options: "Multiclass", "All_and_One", "Clamping", "Bending", "Mandrel Extraction", "De-Clamping".
TARGET_LABEL = "Multiclass"

# Sensor subset used by the classifier pipeline.
# Options: "machine_and_movement", "movement".
TARGET_PROCESS_PART = "machine_and_movement"

# ========================================
# Training Parameters
# ========================================
# Whether to train a new model or only load an existing checkpoint.
# Options: True, False.
ENABLE_MODEL_TRAINING = True

# Batch size used for train/validation/test data loaders.
TRAINING_BATCH_SIZE = 8

# LSTM hidden state size.
LSTM_HIDDEN_SIZE = 64

# Number of stacked LSTM layers.
LSTM_LAYER_COUNT = 2

# Maximum number of training epochs.
MAX_TRAINING_EPOCHS = 100

# Optimizer learning rate.
TRAINING_LEARNING_RATE = 1e-4

# Early-stopping patience in epochs.
EARLY_STOPPING_PATIENCE = 4

# Random seed used for data split, model initialization, and dataloader shuffling.
RANDOM_SEED = 42

# ========================================
# Inference Parameters
# ========================================
# Labels used during one-label-per-model inference.
# Options: any subset/order of "Clamping", "Bending", "Mandrel Extraction", "De-Clamping".

INFERENCE_TARGET_LABELS = [
    "Clamping",
    "Bending",
    "Mandrel Extraction",
    "De-Clamping",
]

ALL_AND_ONE_TRAINING_LABELS = [
    "Clamping",
    "De-Clamping",
    "Mandrel Extraction",
    "Bending",
    "Multiclass",
]

def build_training_pipeline_config() -> dict:
    return {
        "dataloader_config": {
            "batch_size": TRAINING_BATCH_SIZE,
        },
        "model_config": {
            "hidden_size": LSTM_HIDDEN_SIZE,
            "num_layers": LSTM_LAYER_COUNT,
        },
        "training_config": {
            "training": ENABLE_MODEL_TRAINING,
            "num_epochs": MAX_TRAINING_EPOCHS,
            "learning_rate": TRAINING_LEARNING_RATE,
            "patience": EARLY_STOPPING_PATIENCE,
        },
    }

def get_training_labels() -> list[str]:
    if TARGET_LABEL == "All_and_One":
        return ALL_AND_ONE_TRAINING_LABELS
    return [TARGET_LABEL]


def run_training_for_label(label: str):
    logger.info("Starting training for label: %s", label)
    return training_pipeline(
        model_path_root=RESULTS_DIRECTORY,
        database_path=PROCESSED_DATA_DIRECTORY,
        annotation_json_path=ANNOTATION_FILE_PATH,
        process_part=TARGET_PROCESS_PART,
        label=label,
        pipeline_config=build_training_pipeline_config(),
        random_seed=RANDOM_SEED,
    )


def run_prediction_plotting(model, sensors_df, feature_cols, validation_loader) -> None:
    if TARGET_LABEL == "All_and_One":
        plot_all_and_one_as_multiclass(
            database_path=str(PROCESSED_DATA_DIRECTORY),
            annotation_json_path=str(ANNOTATION_FILE_PATH),
            results_directory=str(RESULTS_DIRECTORY),
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYER_COUNT,
            labels=INFERENCE_TARGET_LABELS,
            process_part=TARGET_PROCESS_PART,
            get_all_predictions_fn=get_all_predictions,
        )
        return

    plot_predictions_vs_true_annot(
        model,
        getattr(validation_loader, "dataset", None),
        sensors_df,
        feature_cols,
        {
            "store_plots": True,
            "store_plots_path": str(RESULTS_DIRECTORY),
        },
        TARGET_PROCESS_PART,
        TARGET_LABEL,
    )


def run_inference() -> None:
    save_validation_confusion_matrices(
        database_path=PROCESSED_DATA_DIRECTORY,
        annotation_json_path=ANNOTATION_FILE_PATH,
        results_directory=RESULTS_DIRECTORY,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_LAYER_COUNT,
        labels=INFERENCE_TARGET_LABELS,
        process_part=TARGET_PROCESS_PART,
        batch_size=TRAINING_BATCH_SIZE,
    )

    run_default_inference_plots(
        database_path=PROCESSED_DATA_DIRECTORY,
        annotation_json_path=ANNOTATION_FILE_PATH,
        results_directory=RESULTS_DIRECTORY,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_LAYER_COUNT,
        labels=INFERENCE_TARGET_LABELS,
        process_part=TARGET_PROCESS_PART,
        get_all_predictions_fn=get_all_predictions,
        figsize=(15, 10),
    )


def main():
    setup_logging()

    try:
        training_result = None
        for label in get_training_labels():
            training_result = run_training_for_label(label)

        model, sensors_df, validation_loader, _, feature_cols = training_result
    except Exception as e:
        logger.error(f"Error during training pipeline: {e}")
        return

    try:
        run_prediction_plotting(model, sensors_df, feature_cols, validation_loader)
    except Exception as e:
        logger.error(f"Warning: Plotting predictions failed: {e}")

    try:
        run_inference()
    except Exception as e:
        logger.error(
            f"Warning: Inference for one label in one experiment failed: {e}"
        )


if __name__ == "__main__":
    main()
