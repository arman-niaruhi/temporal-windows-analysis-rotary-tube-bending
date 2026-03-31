import copy
import logging
from pathlib import Path

from src.logging.logging_config import setup_logging

from src.pipeline.ml.context_extractor.utils.helpers.seed_utils import (
    enforce_reproducibility,
)
from src.pipeline.ml.context_extractor.utils.data.data_preprocessor import prepare_data
from src.pipeline.ml.context_extractor.utils.training_pipeline_utils import train_model

logger = logging.getLogger(__name__)


# ========================================
# Configuration
# ========================================
# Global runtime settings shared across the pipeline.
GENERAL_SETTING = {
    # Random seed for Python, NumPy, PyTorch, and data splitting.
    # Value: integer, e.g. 42, 123, 2025.
    "seed": 42,
}

# Input data locations and the process-part selection used for training.
INPUT_PATH_PARAMS = {
    # Directory containing the processed ETL sensor tables.
    # Value: string path, e.g. "data/processed".
    "database_path": str(Path("data") / "processed"),
    # Sensor subset to load.
    # Typical values: "All", "Bending".
    "process_part": "All",
    # JSON file with annotation intervals used to create targets.
    # Value: string path to a .json annotation file.
    "annotation_json_path": str(
        Path("data") / "ml" / "machine-and-movement_complete.json"
    ),
}

# Preprocessing controls for normalization, windowing, feature selection, and split definition.
PREPROCESSING_PARAMS = {
    # Seed used inside preprocessing steps such as deterministic splitting/shuffling.
    # Value: integer.
    "random_seed": 42,
    # Whether sensor inputs should be normalized before training.
    # Values: True or False.
    "normalize": True,
    # Feature scaler applied when normalize=True.
    # Typical values: "standard", "minmax".
    "scaler": "standard",
    # Aggregation rule used when resampling or compressing windows.
    # Typical values: "mean", "max", "median".
    "agg_metric": "mean",
    # Whether the temporal signal should be resampled before training.
    # Values: True or False.
    "resample": True,
    # Target number of temporal windows/timesteps after resampling.
    # Value: integer, e.g. 200, 400, 800.
    "window_num": 400,
    # JSON split definition with train/test experiment groups.
    # Value: string path to a split config .json file.
    "split_config_path": str(
        Path("config")
        / "data-split-config"
        / "train_test_split_based_on_column_gp1.json"
    ),
    # Indices of target features to predict from the available target vector.
    # Value: list[int], e.g. [0], [1, 3], [0, 1, 2, 3].
    "feature_indices": [1, 3],
    # Main process annotation timestamps used in plots.
    # Value: ordered list[int] in timestep units.
    "annot_timesteps": [150, 340, 820, 1280],
    # Start/end timestamps for the mandrel extraction interval.
    # Value: list[int], typically two values [start, end].
    "mandrel_extraction_annot_timesteps": [650, 820],
}

# Model architecture and optimization settings for context extraction training.
TRAINING_PARAMS = {
    # Whether to train a new model or resume analysis from an existing MLflow run.
    # Values: True or False.
    "train": False,
    # Root directory for local temporal-pattern-analysis outputs.
    # Value: string path.
    "results_dir": str(Path("results") / "temporal_pattern_analysis"),
    # Model family to instantiate.
    # Typical values: "lstm", "tcn_lstm".
    "model_type": "lstm",
    # Hidden size of the recurrent backbone or shared hidden representation.
    # Value: integer, e.g. 32, 64, 128.
    "hidden_dim": 32,
    # Number of stacked LSTM layers when the model uses an LSTM block.
    # Value: integer >= 1.
    "lstm_layers": 1,
    # Dropout probability used in the model.
    # Value: float in [0, 1], e.g. 0.1, 0.2, 0.3.
    "dropout": 0.2,
    # Optimizer learning rate.
    # Value: positive float, e.g. 1e-3, 2e-4.
    "lr": 2e-4,
    # AdamW weight decay for regularization.
    # Value: non-negative float, e.g. 0.0, 1e-4.
    "weight_decay": 1e-4,
    # Batch size for train and validation loaders.
    # Value: integer, e.g. 8, 16, 32.
    "batch_size": 16,
    # Maximum number of training epochs.
    # Value: integer >= 1.
    "max_epochs": 5,
    # Number of temporal convolution layers when using a TCN-based model.
    # Value: integer >= 1.
    "tcn_layers": 10,
    # Kernel size of the temporal convolution.
    # Value: integer >= 2, commonly 3 or 5.
    "tcn_kernel_size": 3,
    # Directory where trained context extraction models are stored.
    # Value: string path.
    "model_path": str(Path("models") / "context_extraction"),
    # Enable temporal attention in the model.
    # Values: True or False.
    "use_attention": True,
    # Enable feature-wise attention on top of temporal attention.
    # Values: True or False.
    "use_feature_attention": True,
    # Attention mechanism implementation.
    # Typical values: "bahdanau", "mlp".
    "attention_type": "bahdanau",
    # Add an angle embedding for multi-output prediction heads.
    # Values: True or False.
    "use_angle_embedding": True,
    # Embedding dimension used when use_angle_embedding=True.
    # Value: integer, e.g. 8, 16, 32.
    "angle_embedding_dim": 16,
    # Use separate output heads for multiple target features.
    # Values: True or False.
    "split_output_heads": True,
    # Run timestep sensitivity analysis after training.
    # Values: True or False.
    "timestep_sensitivity": False,
    # Hidden layer sizes for the primary output head MLP.
    # Value: list[int], e.g. [128, 64, 32].
    "main_head_hidden_sizes": [128, 64, 32],
    # Hidden layer sizes for the secondary output head MLP.
    # Value: list[int], e.g. [128, 64, 32].
    "secondary_head_hidden_sizes": [128, 64, 32],
    # Per-target loss weights before optional dynamic reweighting.
    # Value: list[float] matching the number of predicted target features -> [main axis, secondary axis].
    "feature_loss_weights": [1.0, 1.0],
    # Gradually transition from initial to end loss weights during warmup.
    # Values: True or False.
    "dynamic_feature_loss_weighting": True,
    # Number of epochs used for the dynamic loss-weight warmup.
    # Value: integer >= 0.
    "dynamic_feature_loss_warmup_epochs": 20,
    # Final per-target loss weights after warmup.
    # Value: list[float] matching the number of target features -> [main axis, secondary axis].
    "dynamic_feature_loss_end_weights": [1.0, 1.0],
    # Loss function used for each predicted target feature.
    # Typical values per entry: "mse", "smoothl1" -> [main axis, secondary axis].
    "feature_loss_types": ["smoothl1", "mse"],
}

# Parameters for the post-training occlusion-based importance analysis.
OCCLUSION_PARAMS = {
    "window_importance_enabled":False,
    # Width of each occluded temporal window analysis.
    # Value: integer >= 1.
    "occlusion_window_size": 5,
    # Step size between successive occlusion windows.
    # Value: integer >= 1.
    "occlusion_stride": 5,
}


def main():
    setup_logging()

    general_setting = copy.deepcopy(GENERAL_SETTING)
    input_path_param = copy.deepcopy(INPUT_PATH_PARAMS)
    preprocessing_param = copy.deepcopy(PREPROCESSING_PARAMS)
    training_params = copy.deepcopy(TRAINING_PARAMS)
    occlusion_params = copy.deepcopy(OCCLUSION_PARAMS)
    preprocessing_info = copy.deepcopy(PREPROCESSING_PARAMS)
    seed = general_setting.get("seed", 42)

    process_part = input_path_param.get("process_part")

    # ============================================================
    # Seeding for reproducibility
    # ============================================================
    enforce_reproducibility(seed=seed)

    # ============================================================
    # Read and preprocess data
    # ============================================================
    (
        X_train,
        Y_train,
        X_test,
        Y_test,
        springbacks_train,
        springbacks_test,
        experiment_configurations_train,
        experiment_configurations_test,
        sensor_names,
        target_feature_names,
        annot_timesteps,
        mandrel_extraction_annot_timesteps,
        normalization_info,
    ) = prepare_data(
        input_path_param=input_path_param,
        preprocessing_param=preprocessing_param,
    )

    train_model(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        springbacks_train=springbacks_train,
        springbacks_test=springbacks_test,
        experiment_configurations_train=experiment_configurations_train,
        experiment_configurations_test=experiment_configurations_test,
        params=training_params,
        occlusion_params=occlusion_params,
        sensor_names=sensor_names,
        target_feature_names=target_feature_names,
        process_part=process_part,
        preprocessing_info=preprocessing_info,
        annot_timesteps=annot_timesteps,
        mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
        target_scaler=(
            normalization_info.get("target_scaler") if normalization_info else None
        ),
        input_path_param=input_path_param,
        general_setting=general_setting,
    )


if __name__ == "__main__":
    main()
