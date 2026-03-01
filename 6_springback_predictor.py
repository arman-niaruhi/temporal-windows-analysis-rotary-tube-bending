import json
import logging

from src.logging.logging_config import setup_logging
from src.pipeline.ml.context_extractor.utils.data.data_preprocessor import prepare_data
from src.pipeline.ml.spring_back_predictior.training import train_model_springback_tcn_lstm, train_model_springback_random_forest
from src.pipeline.ml.context_extractor.utils.data.data_preprocessor import create_data_loaders


logger = logging.getLogger(__name__)

spring_back_config_path = "config/springback-prediction/springback-prediction-config.json"
 
def main():
    # ============================================================
    # Load schema and config and validate the config json entries and extract the required parameters
    # ============================================================
    setup_logging()
    with open(spring_back_config_path, "r") as f:
        config = json.load(f)
    
    input_path_param = config.get("inputPathParams")
    preprocessing_param = config.get("preprocessingParams")
    lstm_training_params = config.get("lstmTrainingParams")
    seed = config.get("generalSetting").get("seed", 42)
        
    (
        X_train, Y_train, X_test, Y_test, springbacks_train, springbacks_test,
        experiment_configurations_train, experiment_configurations_test,
        sensor_names, target_feature_names, annot_timesteps,
        mandrel_extraction_annot_timesteps, normalization_info,
    ) = prepare_data(
            input_path_param=input_path_param,
            preprocessing_param=preprocessing_param,
        )
    
    # Train RF
    train_model_springback_random_forest(
        X_train=X_train,
        X_test=X_test,
        springbacks_train=springbacks_train,
        springbacks_test=springbacks_test,
        sensor_names = sensor_names
    )
    
    # Train LSTM
    train_loader, val_loader, plot_loader = create_data_loaders(
        X_train,
        Y_train,
        X_test,
        Y_test,
        springbacks_train,
        springbacks_test,
        experiment_configurations_train,
        experiment_configurations_test,
        lstm_training_params["batch_size"],
    )
    
    train_model_springback_tcn_lstm( 
    seed=seed,
    model_input_size=X_train.shape[2],
    model_output_size=springbacks_train.shape[2],
    training_params= lstm_training_params,
    springbacks_train = springbacks_train,
    train_loader=train_loader,
    val_loader=val_loader,
    plot_loader=plot_loader,
    )
        
        

if __name__ == "__main__":
    main()
