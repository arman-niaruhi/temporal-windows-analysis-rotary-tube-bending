import logging

from src.logging.logging_config import setup_logging
from src.pipeline.ml.context_extractor.utils.data.data_preprocessor import prepare_data, create_data_loaders
from src.pipeline.ml.spring_back_predictior.training import train_model_springback_tcn_lstm, train_model_springback_random_forest


logger = logging.getLogger(__name__)

# ========================================
# Configuration
# ========================================
# Random seed used to make springback experiments reproducible across runs.
# Value: integer, e.g. 42, 123, 2025.
SEED = 42

# Input data locations and dataset selection used for springback prediction.
INPUT_PATH_PARAMS = {
    # Directory containing the processed ETL sensor tables.
    # Value: string path, e.g. "data/processed".
    "database_path": "data/processed",
    # Sensor subset to load.
    # Typical values: "All", "movement", "machine_and_movement".
    "process_part": "All",
    # JSON annotation file used to derive target windows and metadata.
    # Value: string path to a .json file.
    "annotation_json_path": "data/ml/machine-and-movement_complete.json",
}

# Preprocessing controls for converting raw sensor sequences into model-ready samples.
PREPROCESSING_PARAMS = {
    # Seed used in preprocessing-related random operations.
    # Value: integer.
    "random_seed": 42,
    # Whether sensor inputs should be normalized before training.
    # Values: True or False.
    "normalize": True,
    # Scaling strategy applied when normalize=True.
    # Typical values: "standard", "minmax".
    "scaler": "standard",
    # Whether to resample each temporal sequence before feature extraction.
    # Values: True or False.
    "resample": False,
    # Aggregation metric used when preprocessing creates fixed-size windows.
    # Typical values: "mean", "max", "median".
    "agg_metric": "mean",
    # Number of timesteps or windows kept per sample after preprocessing.
    # Value: integer, e.g. 200, 400, 800.
    "window_num": 400,
    # Train/test split definition used for the springback experiments.
    # Value: string path to a split config .json file.
    "split_config_path": "config/data-split-config/train_test_split_each_setup_80.json",
    # Indices of target features selected from the available target vector.
    # Value: list[int], e.g. [0], [1, 3], [0, 1, 2, 3].
    "feature_indices": [1, 3],
    # Main process annotation timestamps used for aligned plots and windows.
    # Value: ordered list[int] in timestep units.
    "annot_timesteps": [150, 340, 820, 1280],
    # Start/end timestamps of the mandrel extraction interval.
    # Value: list[int], typically two values [start, end].
    "mandrel_extraction_annot_timesteps": [650, 820],
}

# Hyperparameters for the hybrid TCN-LSTM springback regressor.
LSTM_TRAINING_PARAMS = {
    # Output channels per temporal convolution block.
    # Value: list[int], e.g. [32, 64, 64].
    "tcn_channels": [32, 64, 64],
    # Kernel size of the temporal convolution layers.
    # Value: integer >= 2, commonly 3, 5, or 10.
    "tcn_kernel_size": 10,
    # Dropout used inside the TCN backbone.
    # Value: float in [0, 1].
    "tcn_dropout": 0.1,
    # Pooling strategy after temporal feature extraction.
    # Typical values: "mean", "max".
    "pool": "mean",
    # Whether to train a new model or reuse an existing one if the pipeline supports it.
    # Values: True or False.
    "train": True,
    # Hidden size of the LSTM block.
    # Value: integer, e.g. 32, 64, 128.
    "hidden_size": 32,
    # Number of stacked LSTM layers.
    # Value: integer >= 1.
    "num_layers": 1,
    # Dropout inside the LSTM stack.
    # Value: float in [0, 1].
    "dropout": 0.1,
    # Dropout applied before the final fully connected layers.
    # Value: float in [0, 1].
    "fc_dropout": 0.1,
    # Whether the LSTM should run bidirectionally.
    # Values: True or False.
    "bidirectional": False,
    # Optimizer learning rate.
    # Value: positive float, e.g. 1e-3, 3e-5.
    "lr": 3e-5,
    # Weight decay used by the optimizer.
    # Value: non-negative float.
    "weight_decay": 3e-5,
    # Batch size for the train and validation loaders.
    # Value: integer, e.g. 8, 16, 32.
    "batch_size": 16,
    # Maximum number of training epochs.
    # Value: integer >= 1.
    "max_epochs": 1000,
    # Minimum validation improvement required to reset early stopping.
    # Value: non-negative float, e.g. 1e-4.
    "stop_early_min_delta": 1e-4,
    # Number of epochs without improvement before early stopping.
    # Value: integer >= 1.
    "stop_early_patience": 20,
    # Factor applied when reducing the learning rate on plateau.
    # Value: float in (0, 1), e.g. 0.5.
    "schedular_factor": 0.5,
    # Number of plateau epochs before reducing the learning rate.
    # Value: integer >= 1.
    "schedular_patience": 3,
    # Maximum gradient norm for gradient clipping.
    # Value: positive float, e.g. 1.0 or 5.0.
    "gradient_clip": 1.0,
    # Logging frequency in epochs.
    # Value: integer >= 1.
    "verbose_every": 2,
    # Directory where trained springback models are stored.
    # Value: string path.
    "model_path": "models/spring_back",
}


def main():
    setup_logging()

    # Prepare the training and test sets for springback prediction from the
    # preprocessed tube-bending experiments.
    (
        X_train, Y_train, X_test, Y_test, springbacks_train, springbacks_test,
        experiment_configurations_train, experiment_configurations_test,
        sensor_names, _, _, _, normalization_info,
    ) = prepare_data(
        input_path_param=INPUT_PATH_PARAMS,
        preprocessing_param=PREPROCESSING_PARAMS,
    )

    # Train conventional machine-learning baselines on the same split to enable
    # direct comparison with the sequence model.
    train_model_springback_random_forest(
        X_train=X_train,
        X_test=X_test,
        springbacks_train=springbacks_train,
        springbacks_test=springbacks_test,
        sensor_names=sensor_names,
        normalization_info=normalization_info,
    )

    # Create mini-batch loaders for the sequence model. The plot loader is used
    # for the final evaluation and result visualization after training.
    train_loader, val_loader, plot_loader = create_data_loaders(
        X_train,
        Y_train,
        X_test,
        Y_test,
        springbacks_train,
        springbacks_test,
        experiment_configurations_train,
        experiment_configurations_test,
        LSTM_TRAINING_PARAMS["batch_size"],
    )

    # Train the hybrid TCN-LSTM regressor for springback prediction.
    train_model_springback_tcn_lstm(
        seed=SEED,
        model_input_size=X_train.shape[2],
        model_output_size=springbacks_train.shape[2],
        training_params=LSTM_TRAINING_PARAMS,
        springbacks_train=springbacks_train,
        train_loader=train_loader,
        val_loader=val_loader,
        plot_loader=plot_loader,
        normalization_info=normalization_info,
        model_name="tcn_lstm",
    )


if __name__ == "__main__":
    main()
