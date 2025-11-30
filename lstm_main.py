from src.pipeline.ml.context_extractor.utils.lstm_utils.config.seed import enforce_reproducibility
from src.pipeline.ml.context_extractor.utils.lstm_utils.data.data_preprocessor import prepare_data
from src.pipeline.ml.context_extractor.utils.lstm_utils.models.training import train_model
import json

if __name__ == "__main__":
    with open("src/pipeline/ml/context_extractor/utils/lstm_utils/config/lstm_config.json", "r") as f:
        config = json.load(f)
    enforce_reproducibility(seed=config.get("seed", 42))
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
