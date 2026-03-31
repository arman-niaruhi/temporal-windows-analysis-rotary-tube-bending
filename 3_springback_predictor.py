import logging

from src.logging.logging_config import setup_logging
from src.pipeline.ml.context_extractor.utils.data.data_preprocessor import prepare_data, create_data_loaders
from src.pipeline.ml.spring_back_predictior.training import train_model_springback_tcn_lstm, train_model_springback_random_forest


logger = logging.getLogger(__name__)

# Fixed seed to make the springback experiments reproducible across runs.
SEED = 42

# Paths to the processed tube-geometry dataset and annotation metadata used
# for springback target generation.
INPUT_PATH_PARAMS = {
    "database_path": "data/processed",
    "process_part": "All",
    "annotation_json_path": "data/ml/machine-and-movement_complete.json",
}

# Preprocessing configuration for converting raw time-series measurements into
# model-ready samples. The selected feature indices and annotated timesteps
# define the sensor channels and process windows used in the study.
PREPROCESSING_PARAMS = {
    "random_seed": 42,
    "normalize": True,
    "scaler": "standard",
    "resample": False,
    "agg_metric": "mean",
    "window_num": 400,
    "split_config_path": "config/data-split-config/train_test_split_each_setup_80.json",
    "feature_indices": [1, 3],
    "annot_timesteps": [150, 340, 820, 1280],
    "mandrel_extraction_annot_timesteps": [650, 820],
}

# Hyperparameters of the hybrid TCN-LSTM model used for springback regression.
# These values define the temporal convolution backbone, recurrent head, and
# optimization settings reported for the learning-based baseline.
LSTM_TRAINING_PARAMS = {
    "tcn_channels": [32, 64, 64],
    "tcn_kernel_size": 10,
    "tcn_dropout": 0.1,
    "pool": "mean",
    "train": True,
    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.1,
    "fc_dropout": 0.1,
    "bidirectional": False,
    "lr": 3e-5,
    "weight_decay": 3e-5,
    "batch_size": 16,
    "max_epochs": 1000,
    "stop_early_min_delta": 1e-4,
    "stop_early_patience": 20,
    "schedular_factor": 0.5,
    "schedular_patience": 3,
    "gradient_clip": 1.0,
    "verbose_every": 2,
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
