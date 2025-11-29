import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
)
import torch


def compute_all_metrics(y_true, y_pred):
    """Compute comprehensive metrics"""
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    mse = mean_squared_error(y_true_np.flatten(), y_pred_np.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_np.flatten(), y_pred_np.flatten())
    r2 = r2_score(y_true_np.flatten(), y_pred_np.flatten())

    per_pred_mse = np.mean((y_true_np - y_pred_np) ** 2, axis=(0, 2))
    per_pred_mae = np.mean(np.abs(y_true_np - y_pred_np), axis=(0, 2))

    if y_true_np.ndim == 3 and y_true_np.shape[2] > 1:
        per_feature_mse = np.mean((y_true_np - y_pred_np) ** 2, axis=(0, 1))
        per_feature_mae = np.mean(np.abs(y_true_np - y_pred_np), axis=(0, 1))
    else:
        per_feature_mse = None
        per_feature_mae = None

    max_error = np.max(np.abs(y_true_np - y_pred_np))
    mean_error = np.mean(y_pred_np - y_true_np)
    std_error = np.std(y_pred_np - y_true_np)
    residuals = y_pred_np - y_true_np

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "max_error": float(max_error),
        "mean_error": float(mean_error),
        "std_error": float(std_error),
        "per_prediction_mse": per_pred_mse.tolist(),
        "per_prediction_mae": per_pred_mae.tolist(),
        "residuals": residuals,
    }

    if per_feature_mse is not None:
        metrics["per_feature_mse"] = per_feature_mse.tolist()
        metrics["per_feature_mae"] = per_feature_mae.tolist()

    return metrics


def compute_epoch_metrics(y_true, y_pred):
    """Compute comprehensive metrics for a single epoch"""
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    # Flatten for overall metrics
    y_true_flat = y_true_np.flatten()
    y_pred_flat = y_pred_np.flatten()

    # Basic metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)

    # Additional metrics
    rmse = np.sqrt(mse)

    epsilon = 1e-8
    # Use mean of absolute values as denominator to avoid division by near-zero
    denominator = np.abs(y_true_flat) + epsilon
    # Alternative: use max(absolute_value, epsilon) for each element
    # denominator = np.maximum(np.abs(y_true_flat), epsilon)

    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / denominator)) * 100

    # Max Error
    max_error = np.max(np.abs(y_true_flat - y_pred_flat))

    # Explained Variance Score (similar to R² but different calculation)
    from sklearn.metrics import explained_variance_score

    evs = explained_variance_score(y_true_flat, y_pred_flat)

    # Mean Bias Error (shows if model systematically over/under predicts)
    mbe = np.mean(y_pred_flat - y_true_flat)

    # Median Absolute Error (more robust to outliers than MAE)
    from sklearn.metrics import median_absolute_error

    medae = median_absolute_error(y_true_flat, y_pred_flat)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "max_error": max_error,
        "evs": evs,
        "mbe": mbe,
        "medae": medae,
    }
