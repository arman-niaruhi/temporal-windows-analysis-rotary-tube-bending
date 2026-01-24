import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
)
import torch

from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error


def compute_all_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute a comprehensive set of metrics for model evaluation.

    Supports per-prediction, per-feature, and overall metrics.
    Converts torch tensors to numpy arrays if necessary.

    Args:
        y_true: Ground truth values (torch tensor or numpy array)
        y_pred: Predicted values (torch tensor or numpy array)

    Returns:
        dict: Contains metrics including MSE, RMSE, MAE, R2, max_error,
              mean_error, std_error, per-prediction/per-feature metrics,
              and residuals.
    """

    # Ensure numpy arrays for metric computation
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    # Overall metrics
    mse = mean_squared_error(y_true_np.flatten(), y_pred_np.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_np.flatten(), y_pred_np.flatten())
    r2_flat = r2_score(y_true_np.flatten(), y_pred_np.flatten())

    r2_uniform_avg = None
    r2_variance_weighted = None
    if y_true_np.ndim == 3:
        y_true_2d = y_true_np.reshape(-1, y_true_np.shape[2])
        y_pred_2d = y_pred_np.reshape(-1, y_pred_np.shape[2])
        r2_uniform_avg = r2_score(y_true_2d, y_pred_2d, multioutput="uniform_average")
        r2_variance_weighted = r2_score(y_true_2d, y_pred_2d, multioutput="variance_weighted")

    # Per-prediction metrics (mean across batch and time)
    per_pred_mse = np.mean((y_true_np - y_pred_np) ** 2, axis=(0, 2))
    per_pred_mae = np.mean(np.abs(y_true_np - y_pred_np), axis=(0, 2))

    # Per-feature metrics (if multi-feature output)
    if y_true_np.ndim == 3:
        per_feature_mse = np.mean((y_true_np - y_pred_np) ** 2, axis=(0, 1))
        per_feature_mae = np.mean(np.abs(y_true_np - y_pred_np), axis=(0, 1))
        per_feature_r2 = [
            r2_score(y_true_np[:, :, i].flatten(), y_pred_np[:, :, i].flatten())
            for i in range(y_true_np.shape[2])
        ]
    else:
        per_feature_mse = None
        per_feature_mae = None
        per_feature_r2 = None

    # Per-sample MSE (average over crosscuts/features)
    if y_true_np.ndim == 3:
        per_sample_mse = np.mean((y_true_np - y_pred_np) ** 2, axis=(1, 2))
    else:
        per_sample_mse = np.mean((y_true_np - y_pred_np) ** 2, axis=1)

    # Additional error statistics
    max_error = np.max(np.abs(y_true_np - y_pred_np))
    mean_error = np.mean(y_pred_np - y_true_np)
    std_error = np.std(y_pred_np - y_true_np)
    residuals = y_pred_np - y_true_np

    # Aggregate metrics in dictionary
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2_uniform_avg) if r2_uniform_avg is not None else float(r2_flat),
        "r2_flat": float(r2_flat),
        "r2_variance_weighted": float(r2_variance_weighted) if r2_variance_weighted is not None else None,
        "max_error": float(max_error),
        "mean_error": float(mean_error),
        "std_error": float(std_error),
        "per_prediction_mse": per_pred_mse.tolist(),
        "per_prediction_mae": per_pred_mae.tolist(),
        "residuals": residuals,
        "per_sample_mse": per_sample_mse.tolist(),
    }

    # Include per-feature metrics if applicable
    if per_feature_mse is not None:
        metrics["per_feature_mse"] = per_feature_mse.tolist()
        metrics["per_feature_mae"] = per_feature_mae.tolist()
        metrics["per_feature_r2"] = per_feature_r2

    return metrics


def compute_epoch_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute standard regression metrics for a single epoch.

    Metrics include:
    - MSE, RMSE, MAE, R2, MAPE
    - Max error, Explained Variance Score (EVS)
    - Mean bias error (MBE), Median Absolute Error (MedAE)

    Args:
        y_true: Ground truth values (torch tensor or numpy array)
        y_pred: Predicted values (torch tensor or numpy array)

    Returns:
        dict: Dictionary of computed metrics
    """

    # Convert tensors to numpy arrays
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    # Flatten for overall metrics
    y_true_flat = y_true_np.flatten()
    y_pred_flat = y_pred_np.flatten()

    # Basic regression metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2_flat = r2_score(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error (avoid division by zero by masking near-zero targets)
    epsilon = 1e-8
    valid_mask = np.abs(y_true_flat) > epsilon
    if np.any(valid_mask):
        mape = np.mean(np.abs((y_true_flat[valid_mask] - y_pred_flat[valid_mask]) / y_true_flat[valid_mask])) * 100
    else:
        mape = np.nan

    # Maximum absolute error
    max_error = np.max(np.abs(y_true_flat - y_pred_flat))

    # Explained Variance Score
    evs = explained_variance_score(y_true_flat, y_pred_flat)

    # Mean Bias Error
    mbe = np.mean(y_pred_flat - y_true_flat)

    # Median Absolute Error
    medae = median_absolute_error(y_true_flat, y_pred_flat)

    # Per-feature metrics
    r2_uniform_avg = None
    r2_variance_weighted = None
    if y_true_np.ndim == 3:
        per_feature_mse = np.mean((y_true_np - y_pred_np) ** 2, axis=(0, 1))
        per_feature_r2 = [
            r2_score(y_true_np[:, :, i].flatten(), y_pred_np[:, :, i].flatten())
            for i in range(y_true_np.shape[2])
        ]
        y_true_2d = y_true_np.reshape(-1, y_true_np.shape[2])
        y_pred_2d = y_pred_np.reshape(-1, y_pred_np.shape[2])
        r2_uniform_avg = r2_score(y_true_2d, y_pred_2d, multioutput="uniform_average")
        r2_variance_weighted = r2_score(y_true_2d, y_pred_2d, multioutput="variance_weighted")
        per_sample_mse = np.mean((y_true_np - y_pred_np) ** 2, axis=(1, 2))
    else:
        per_feature_mse = None
        per_feature_r2 = None
        per_sample_mse = np.mean((y_true_np - y_pred_np) ** 2, axis=1)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2_uniform_avg if r2_uniform_avg is not None else r2_flat,
        "r2_flat": r2_flat,
        "r2_variance_weighted": r2_variance_weighted,
        "mape": mape,
        "max_error": max_error,
        "evs": evs,
        "mbe": mbe,
        "medae": medae,
        "per_feature_mse": per_feature_mse.tolist() if per_feature_mse is not None else None,
        "per_feature_r2": per_feature_r2,
        "per_sample_mse": per_sample_mse.tolist(),
        "per_sample_mse_mean": float(np.mean(per_sample_mse)),
    }
