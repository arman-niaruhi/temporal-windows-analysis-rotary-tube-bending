import torch
import torch.nn as nn
from src.pipeline.ml.context_extractor.utils.helpers.metrics_utils import compute_all_metrics
import shutil
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# MLflow experiment setup
# ============================================================
def setup_mlflow_experiment(process_part: str, params: dict,
                            preprocessing_info: dict, X: torch.Tensor,
                            Y: torch.Tensor, target_feature_names: list[str],
                            input_path_param: dict | None = None,
                            general_setting: dict | None = None) -> tuple:
    """
    Initialize MLflow experiment and generate experiment description.

    Args:
        process_part: Name of the machine part for labeling the experiment
        params: Dictionary of model hyperparameters
        preprocessing_info: Dictionary describing preprocessing applied
        X: Input tensor (N_EXPERIMENTS x TIMESTEPS x FEATURES_IN)
        Y: Output tensor (N_EXPERIMENTS x N_CROSSCUT x FEATURES_OUT)
        target_feature_names: List of feature names being predicted

    Returns:
        tuple: (experiment_description, FEATURES_IN, N_CROSSCUT, FEATURES_OUT)
    """
    # End any currently active MLflow run
    if mlflow.active_run() is not None:
        mlflow.end_run()
    
    # Create experiment
    mlflow.set_experiment("LSTM_Attention-All")
    mlflow.set_tracking_uri(str(Path("mlruns").resolve()))
    
    N_EXPERIMENTS, TIMESTEPS_IN, FEATURES_IN = X.shape
    N_EXPERIMENTS, N_CROSSCUT, FEATURES_OUT = Y.shape
    
    split_config_path = preprocessing_info.get(
        "split_config_path",
        "config/data-split-config/train_test_split.json",
    )
    model_type = params.get("model_type", "unknown")
    # Detailed experiment description for reproducibility/logging
    experiment_description = f"""
    SPLIT CONFIG: {split_config_path}
    PART: {process_part}
    MODEL TYPE: {model_type}
    ================== GENERAL SETTINGS ==================
    {general_setting}
    ==================== INPUT PATHS =====================
    {input_path_param}
    ============== PREPROCESSING INFO ====================
    {preprocessing_info}
    ==================== MODEL INFO ======================
    INPUT OF TRAIN:
    N_EXPERIMENTS, N_TIMESTEPS_IN, N_FEATURES_IN = ({N_EXPERIMENTS}, {TIMESTEPS_IN}, {FEATURES_IN})
    OUTPUT_VALIDATION:
    N_EXPERIMENTS, N_CROSSCUT, FEATURES_OUT = ({N_EXPERIMENTS}, {N_CROSSCUT}, {FEATURES_OUT})
   
    GEOMETRY_FEATURES: {target_feature_names}
    ================== TRAINING INFO =====================
    {params}
    """
    return experiment_description, FEATURES_IN, N_CROSSCUT, FEATURES_OUT


def _flatten_dict(data: dict, prefix: str) -> dict:
    if data is None:
        return {}
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def log_scalar_metrics_from_config(config: dict, prefix: str = "config.") -> None:
    """
    Log scalar config values as MLflow metrics.
    Only int/float/bool values are logged; lists and dicts are skipped.
    """
    def _collect_scalars(data: dict, key_prefix: str) -> dict:
        metrics: dict[str, float] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                metrics.update(_collect_scalars(value, f"{key_prefix}{key}."))
            elif isinstance(value, bool):
                metrics[f"{key_prefix}{key}"] = float(value)
            elif isinstance(value, (int, float)):
                metrics[f"{key_prefix}{key}"] = float(value)
        return metrics

    metrics = _collect_scalars(config, prefix)
    if metrics:
        mlflow.log_metrics(metrics)


def log_experiment_metadata_to_mlflow(
    *,
    split_config_path: str | None,
    process_part: str,
    model_type: str,
    general_setting: dict | None,
    input_path_param: dict | None,
    preprocessing_info: dict,
    params: dict,
) -> None:
    flat = {}
    if split_config_path:
        flat["split_config_path"] = split_config_path
    flat["process_part"] = process_part
    flat["model_type"] = model_type
    flat.update(_flatten_dict(general_setting, "general."))
    flat.update(_flatten_dict(input_path_param, "input."))
    flat.update(_flatten_dict(preprocessing_info, "preprocess."))
    flat.update(_flatten_dict(params, "train."))
    mlflow.log_params(flat)


# ============================================================
# Model logging
# ============================================================
def log_model_parameters(model: nn.Module) -> None:
    """Log total and trainable parameters of the model to MLflow."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlflow.log_param("total_parameters", total_params)
    mlflow.log_param("trainable_parameters", trainable_params)


# ============================================================
# Epoch-level metrics logging
# ============================================================
def log_epoch_metrics(epoch: int, train_loss: float, val_loss: float, 
                      metrics: dict, current_lr: float, epoch_time: float) -> None:
    """
    Log metrics of a single training epoch to MLflow.

    Includes loss values, regression metrics, learning rate, and epoch duration.
    """
    metrics_to_log = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_mse": metrics["mse"],
        "val_rmse": metrics["rmse"],
        "val_mae": metrics["mae"],
        "val_r2": metrics["r2"],
        "val_mape": metrics["mape"],
        "val_max_error": metrics["max_error"],
        "val_evs": metrics["evs"],
        "val_mbe": metrics["mbe"],
        "val_medae": metrics["medae"],
        "val_sample_mse_mean": metrics.get("per_sample_mse_mean"),
        "learning_rate": current_lr,
        "epoch_time": epoch_time,
    }

    per_feature_mse = metrics.get("per_feature_mse") or []
    per_feature_r2 = metrics.get("per_feature_r2") or []
    for i, mse in enumerate(per_feature_mse):
        metrics_to_log[f"val_mse_feature_{i}"] = mse
    for i, r2 in enumerate(per_feature_r2):
        metrics_to_log[f"val_r2_feature_{i}"] = r2

    mlflow.log_metrics(metrics_to_log, step=epoch)


# ============================================================
# Final metrics logging
# ============================================================
def log_final_metrics(all_targets: torch.Tensor, all_preds: torch.Tensor, 
                      val_losses: list, epoch_times: list) -> None:
    """
    Compute and log final metrics after training.

    Logs overall regression metrics and per-feature metrics if applicable.
    """
    final_metrics = compute_all_metrics(all_targets, all_preds)
    
    metrics_to_log = {
        "final_mse": final_metrics["mse"],
        "final_rmse": final_metrics["rmse"],
        "final_mae": final_metrics["mae"],
        "final_r2": final_metrics["r2"],
        "final_max_error": final_metrics["max_error"],
        "final_mean_error": final_metrics["mean_error"],
        "final_std_error": final_metrics["std_error"],
        "total_epochs": len(val_losses),
        "avg_epoch_time": np.mean(epoch_times),
    }

    # Log per-feature metrics if they exist
    if "per_feature_mse" in final_metrics:
        for i, (mse, mae) in enumerate(zip(final_metrics["per_feature_mse"], final_metrics["per_feature_mae"])):
            metrics_to_log[f"final_mse_feature_{i}"] = mse
            metrics_to_log[f"final_mae_feature_{i}"] = mae
        for i, r2 in enumerate(final_metrics.get("per_feature_r2", [])):
            metrics_to_log[f"final_r2_feature_{i}"] = r2

    if "per_sample_mse" in final_metrics:
        metrics_to_log["final_sample_mse_mean"] = float(np.mean(final_metrics["per_sample_mse"]))

    mlflow.log_metrics(metrics_to_log)

    if "per_sample_mse" in final_metrics:
        df = pd.DataFrame({
            "sample_index": np.arange(len(final_metrics["per_sample_mse"])),
            "mse": final_metrics["per_sample_mse"],
        })
        df.to_csv("final_per_sample_mse.csv", index=False)
        mlflow.log_artifact("final_per_sample_mse.csv")
        Path("final_per_sample_mse.csv").unlink()


# ============================================================
# Save experiment description to MLflow
# ============================================================
def save_experiment_description_as_text(description: str) -> None:
    """Save experiment description as a text file in MLflow artifacts."""
    desc_path = Path("experiment_description.txt")
    with open(desc_path, "w") as f:
        f.write(description)
    mlflow.log_artifact(str(desc_path))
    desc_path.unlink()  # remove local file
    logger.info("Experiment description saved to MLflow artifacts.")


# ============================================================
# Best model tracking
# ============================================================
def update_best_model(val_loss: float, best_val_loss: float, model: nn.Module,
                      patience: int, epoch: int) -> tuple:
    """
    Compare current validation loss with best so far.

    Returns updated best_val_loss, best_state, and patience counter.
    """
    if val_loss < best_val_loss - 1e-6:  # improvement threshold
        best_val_loss = val_loss
        best_state = model.state_dict()
        patience = 0
        mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
    else:
        patience += 1
        best_state = None
    
    return best_val_loss, best_state, patience


# ============================================================
# Feature importance logging
# ============================================================
def log_feature_importance_to_mlflow(combined_importance_df: dict) -> None:
    """
    Log feature importance results as a CSV to MLflow artifacts.
    """
    if combined_importance_df is None:
        return
    
    # Ensure all values are lists
    combined_importance_df = {k: v if isinstance(v, list) else [v] for k, v in combined_importance_df.items()}
    df = pd.DataFrame(combined_importance_df)
    df.to_csv("feature_importance_summary.csv", index=False)
    mlflow.log_artifact("feature_importance_summary.csv")
    Path("feature_importance_summary.csv").unlink()


# ============================================================
# Move all generated images to MLflow artifacts
# ============================================================
def move_images_to_mlflow_artifacts(images_dir_path) -> None | bool:
    """
    Move a directory of images to MLflow artifacts and remove local copies.
    """
    try:
        run = mlflow.active_run()
        if run is None:
            logger.warning("No active MLflow run found. Cannot log images to MLflow artifacts.")
            return None

        if images_dir_path.exists():
            mlflow.log_artifact(str(images_dir_path))
            shutil.rmtree(images_dir_path)  # remove local copy
            return True
        else:
            logger.warning(f"Image directory {images_dir_path} does not exist. Skipping MLflow logging.")
            return None

    except Exception as e:
        logger.error(f"Error logging images to MLflow: {e}")
        return None


# ============================================================
# Retrieve previous MLflow run
# ============================================================
def find_previous_mlflow_run(process_part: str, preprocessing_info: dict):
    """
    Search for the most recent MLflow run matching a naming pattern.

    Returns:
        tuple: (run_id, model_uri) if found, else (None, None)
    """
    client = MlflowClient()
    experiment_name = "LSTM_Attention-All"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"No MLflow experiment found with name {experiment_name}.")
        return None, None

    # Construct run name from process_part and preprocessing info
    excluded58 = "" if not preprocessing_info.get('to_58_excluded', False) else "58"
    window_size = str(preprocessing_info.get('window_num', '0'))
    run_name_to_search = f"{process_part}_{excluded58}_ws{window_size}"

    # Search runs by run_name tag
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name_to_search}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if runs:
        run = runs[0]
        logger.info(f"Found previous run: {run.info.run_name} (run_id: {run.info.run_id})")
        model_uri = f"runs:/{run.info.run_id}/model"
        return run.info.run_id, model_uri

    logger.info(f"No previous run found with run_name: {run_name_to_search}")
    return None, None
