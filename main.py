import argparse
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import json
import os

from src.logging.logging_config import setup_logging
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
from src.pipeline.ml.context_extractor.utils.helpers.seed_utils import enforce_reproducibility
from src.pipeline.ml.context_extractor.utils.data.data_preprocessor import prepare_data
from src.pipeline.ml.context_extractor.utils.training_pipeline_utils import train_model

from src.pipeline.ml.spring_back_predictior.training import train_model_springback_lstm, train_model_springback_random_forest
from src.pipeline.ml.context_extractor.utils.data.data_preprocessor import create_data_loaders


import logging

logger = logging.getLogger(__name__)


activity_recognition_config_path = "config/machine-activity-recognition/machine-activity-recognition-config.json"
activity_recognition_config_schema_path = "config/machine-activity-recognition/machine-activity-recognition-config-schema.json"
        
context_extraction_config_path = "config/context-extraction/context-extraction-config.json"
context_extraction_config_schema_path = "config/context-extraction/context-extraction-config-schema.json"

spring_back_config_path = "config/springback-prediction/springback-prediction-config.json"

def get_schema_description(schema_part, path):
            """Traverse schema to find 'description' for a given path."""
            for key in path:
                if 'properties' in schema_part and key in schema_part['properties']:
                    schema_part = schema_part['properties'][key]
                else:
                    return None
            return schema_part.get('description')
         
def main():
    # ============================================================
    # Load schema and config and validate the config json entries and extract the required parameters
    # ============================================================
    setup_logging()
    parser = argparse.ArgumentParser(description="Run different pipeline steps.")
    parser.add_argument(
        "step",
        choices=["preprocess", "activity_recognition", "context_extraction", "springback_prediction"],
        help="Choose which pipeline step to run",
    )
    args = parser.parse_args()

    if args.step == "preprocess":
        with open("config/preprocessing/preprocessing_config.json", "r") as f:
            config = json.load(f)
        DataPreprocessPipeline.run(
            failed_experiment=config["failed_experiment"],
            eliminated_columns=config["eliminated_columns"],
            normalized_tables=config["normalized_tables"],
            nan_handler=True,
            correlation_matrices=config["correlation_matrices"],
        )

    elif args.step == "activity_recognition":
        # ============================================================
        # Load schema and config and validate the config json entries and extract the required parameters
        # ============================================================

        with open(activity_recognition_config_path, "r") as f:
            config = json.load(f)
        with open(activity_recognition_config_schema_path, "r") as f:
            schema = json.load(f)

        try:
            validate(instance=config, schema=schema)
            logging.info("Configuration json is valid!")
        except ValidationError as e:
            path_list = list(e.path)
            description = get_schema_description(schema, path_list)
            
            if description:
                message = f"Validation failed at '{' -> '.join(str(p) for p in path_list)}': {description}"
            else:
                message = f"Validation failed at '{' -> '.join(str(p) for p in path_list)}': {e.message}"
            
            logging.error(message)
            
                
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
                model, sensors_df, test_loader, device, feature_cols = (
                    training_pipeline(
                        model_path_root=config.get("model_path_root"),
                        database_path=config.get("database_path"),
                        annotation_json_path=config.get("annotation_json_path"),
                        experiment_ids_path=config.get("experiment_ids_path"),
                        process_part=config.get("process_part"),
                        eliminated_columns=config.get("eliminated_columns"),
                        label=config.get("label"),
                        pipeline_config=config.get("pipeline_config"),
                    )
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
                analyze_features(analyze_features_result_path, model, sensors_df, test_loader, device)
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
                TEST_EXPERIMENT_IDS = [
                    2, 3, 22, 23, 40, 54, 83, 85, 110, 112, 119, 120, 121, 122, 123,
                    178, 179, 182, 183, 211, 212, 213, 255, 258, 261, 271, 272, 273,
                    302, 303, 304, 317, 318
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

    elif args.step == "context_extraction":
        # ============================================================
        # Load schema and config and validate the config json entries and extract the required parameters
        # ============================================================
        with open(context_extraction_config_path, "r") as f:
            config = json.load(f)
        with open(context_extraction_config_schema_path, "r") as f:
            schema = json.load(f)

        try:
            validate(instance=config, schema=schema)
            logging.info("Configuration json is valid!")
        except ValidationError as e:
            path_list = list(e.path)
            description = get_schema_description(schema, path_list)
            
            if description:
                message = f"Validation failed at '{' -> '.join(str(p) for p in path_list)}': {description}"
            else:
                message = f"Validation failed at '{' -> '.join(str(p) for p in path_list)}': {e.message}"
            
            logging.error(message)
        
        
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
        try:
            X_train, Y_train, X_test, Y_test, springbacks_train, springbacks_test, sensor_names, target_feature_names, annot_timesteps, mandrel_extraction_annot_timesteps = prepare_data(
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
                params=training_params,
                occlusion_params=occlusion_params,
                sensor_names=sensor_names,
                target_feature_names=target_feature_names,
                process_part=process_part,
                preprocessing_info=preprocessing_info,
                annot_timesteps=annot_timesteps,
                mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
                )
    
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return

    elif args.step == "springback_prediction":
        with open(spring_back_config_path, "r") as f:
            config = json.load(f)
        
        input_path_param = config.get("inputPathParams")
        preprocessing_param = config.get("preprocessingParams")
        lstm_training_params = config.get("lstmTrainingParams")
        seed = config.get("generalSetting").get("seed", 42)
            
        X_train, Y_train, X_test, Y_test, springbacks_train, springbacks_test, sensor_names, target_feature_names, annot_timesteps, mandrel_extraction_annot_timesteps = prepare_data(
                input_path_param=input_path_param,
                preprocessing_param=preprocessing_param,
            )
        train_loader, val_loader, plot_loader = create_data_loaders(
        X_train, Y_train, X_test, Y_test, springbacks_train, springbacks_test, lstm_training_params["batch_size"])
        train_model_springback_random_forest(
            X_train=X_train,
            X_test=X_test,
            springbacks_train=springbacks_train,
            springbacks_test=springbacks_test
        )
        '''
        train_model_springback_lstm( 
        seed=seed,
        model_input_size=X_train.shape[2],
        model_output_size=springbacks_train.shape[2],
        training_params= lstm_training_params,
        springbacks_train = springbacks_train,
        train_loader=train_loader,
        val_loader=val_loader,
        plot_loader=plot_loader,
        )
        '''

if __name__ == "__main__":
    main()
