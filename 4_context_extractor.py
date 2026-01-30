import json
import logging

from src.logging.logging_config import setup_logging

from src.pipeline.ml.context_extractor.utils.helpers.seed_utils import (
    enforce_reproducibility,
)
from src.pipeline.ml.context_extractor.utils.data.data_preprocessor import prepare_data
from src.pipeline.ml.context_extractor.utils.training_pipeline_utils import train_model

logger = logging.getLogger(__name__)


context_extraction_config_path = (
    "config/context-extraction/context-extraction-config.json"
)


def main():

    setup_logging()
    with open(context_extraction_config_path, "r") as f:
        config = json.load(f)

    input_path_param = config.get("inputPathParams")
    preprocessing_param = config.get("preprocessingParams")
    training_params = config.get("trainingParams")
    occlusion_params = config.get("occlusionParams")
    preprocessing_info = config.get("preprocessingParams")
    seed = config.get("generalSetting").get("seed", 42)

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
        general_setting=config.get("generalSetting"),
    )


if __name__ == "__main__":
    main()
