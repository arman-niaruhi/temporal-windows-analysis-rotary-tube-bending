"""Visualization utilities for time series predictions and annotations."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import torch

logger = logging.getLogger(__name__)

  # Test experiments
TEST_EXPERIMENT_IDS = [
    2, 3, 22, 23, 40, 54, 83, 85, 110, 112, 119, 120, 121, 122, 123,
    178, 179, 182, 183, 211, 212, 213, 255, 258, 261, 271, 272, 273,
    302, 303, 304, 317, 318
]


def _predict_experiment(
    model: torch.nn.Module,
    exp_data: pd.DataFrame,
    feature_cols: List[str],
    idx_to_label: Dict[int, str],
    device: str = "cpu",
) -> tuple:
    """Generate predictions for a single experiment.
    
    Args:
        model: Trained PyTorch model
        exp_data: DataFrame for single experiment
        feature_cols: Feature column names
        idx_to_label: Mapping from indices to label names
        device: Device for inference
        
    Returns:
        Tuple of (predicted_labels, true_labels, timestamps)
    """
    model.eval()
    
    # Prepare input
    X = torch.tensor(exp_data[feature_cols].values, dtype=torch.float32).to(device)
    X = X.unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(X)
        y_pred = torch.argmax(outputs, dim=-1).squeeze(0).cpu().numpy()
    
    # Convert to label names
    y_pred_names = [idx_to_label[p] for p in y_pred]
    y_true = exp_data["Label"].values
    y_true_names = [
        idx_to_label[label] if isinstance(label, int) else label 
        for label in y_true
    ]
    timestamps = exp_data.index.astype(float)
    
    return y_pred_names, y_true_names, timestamps


def _plot_experiment(
    exp_id: int,
    exp_data: pd.DataFrame,
    feature_cols: List[str],
    y_pred_names: List[str],
    y_true_names: List[str],
    timestamps: pd.Index,
    save_path: Path,
) -> None:
    """Plot sensor data with predictions and true labels.
    
    Args:
        exp_id: Experiment ID
        exp_data: DataFrame for single experiment
        feature_cols: Feature columns to plot
        y_pred_names: Predicted label names
        y_true_names: True label names
        timestamps: Time index
        save_path: Path to save the plot
    """
    # Create color mapping
    unique_labels = sorted(set(y_pred_names) | set(y_true_names))
    cmap = plt.get_cmap("tab10")
    color_map = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}
    
    # Create figure
    fig, axes = plt.subplots(
        4, 1, 
        figsize=(14, 8), 
        gridspec_kw={"height_ratios": [3, 1, 1, 0.5]}
    )
    ax_data, ax_pred, ax_true, ax_leg = axes
    
    # Plot sensor data
    for col in feature_cols:
        ax_data.plot(timestamps, exp_data[col].values, label=col, linewidth=0.8)
    ax_data.set_title(f"Experiment {exp_id} — Sensor Data & Predictions")
    ax_data.set_ylabel("Sensor Values")
    ax_data.grid(True, alpha=0.3)
    
    # Plot predictions
    for i in range(len(timestamps) - 1):
        t_start, t_end = timestamps[i], timestamps[i + 1]
        ax_pred.axvspan(t_start, t_end, color=color_map[y_pred_names[i]], alpha=0.3)
    ax_pred.set_ylabel("Prediction")
    ax_pred.set_yticks([])
    
    # Plot true labels
    for i in range(len(timestamps) - 1):
        t_start, t_end = timestamps[i], timestamps[i + 1]
        ax_true.axvspan(t_start, t_end, color=color_map[y_true_names[i]], alpha=0.3)
    ax_true.set_ylabel("True Label")
    ax_true.set_xlabel("Time")
    ax_true.set_yticks([])
    
    # Create legend
    patches = [
        mpatches.Patch(color=color, alpha=0.3, label=label)
        for label, color in color_map.items()
    ]
    ax_leg.axis("off")
    ax_leg.legend(handles=patches, loc="center", ncol=5, frameon=False)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_predictions_vs_true_annot(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    sensor_df: pd.DataFrame,
    feature_cols: List[str],
    plot_config: Dict,
    machine_part: str,
    objective_label: str,
    device: str = "cpu",
) -> None:
    """Plot predictions vs true annotations for multiple experiments.
    
    Args:
        model: Trained PyTorch model
        dataset: Dataset with label mappings
        sensor_df: DataFrame with sensor data and labels
        feature_cols: List of feature column names
        plot_config: Dict with 'store_plots' and 'store_plots_path'
        machine_part: Machine part identifier
        objective_label: Objective label
        device: Device for inference
    """
    if not plot_config.get("store_plots", False):
        logger.info("Plot storage disabled, skipping plot generation")
        return
    
    store_path_root = plot_config.get("store_plots_path")
    if not store_path_root:
        logger.error("store_plots_path not specified in plot_config")
        return
    
    output_dir = Path(store_path_root) / machine_part / objective_label
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    
    successful = 0
    for exp_id in TEST_EXPERIMENT_IDS:
        try:
            logger.info(f"Plotting predictions for Experiment ID: {exp_id}")
            
            # Get experiment data
            exp_data = sensor_df[sensor_df["Experiment_ID"] == str(exp_id)]
            if exp_data.empty:
                logger.warning(f"No data found for Experiment ID {exp_id}")
                continue
            
            # Predict
            y_pred_names, y_true_names, timestamps = _predict_experiment(
                model, exp_data, feature_cols, idx_to_label, device
            )
            
            # Plot
            save_path = output_dir / f"labeled_timestamps_{exp_id}.png"
            _plot_experiment(
                exp_id, exp_data, feature_cols, 
                y_pred_names, y_true_names, timestamps, save_path
            )
            
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed to plot Experiment {exp_id}: {e}")
            continue
    
    logger.info(f"Successfully created {successful}/{len(TEST_EXPERIMENT_IDS)} plots")