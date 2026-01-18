"""Visualization utilities for time series predictions and annotations."""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import torch
import numpy as np

logger = logging.getLogger(__name__)

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
    
    X = torch.tensor(exp_data[feature_cols].values, dtype=torch.float32).to(device)
    X = X.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(X)
        y_pred = torch.argmax(outputs, dim=-1).squeeze(0).cpu().numpy()
    
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
    Exact same layout as your example with time indices."""
    
    all_labels = set(y_pred_names) | set(y_true_names)
    labels = sorted([label for label in all_labels if label != 'No Label'])
    
    if not labels:
        print(f"No labels (excluding 'No Label') found for experiment {exp_id}")
        return
    
    base_colors = list(mcolors.TABLEAU_COLORS.values())[:len(labels)]
    
    fig_width = 16
    fig_height = 5
    figsize = (fig_width, fig_height)

    _, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1])

    sensor_plotted = set()
    for col in feature_cols:
        if col not in sensor_plotted:
            axs[0].plot(range(len(exp_data)), exp_data[col].values, linewidth=0.8, label=col)
            sensor_plotted.add(col)
        else:
            axs[0].plot(range(len(exp_data)), exp_data[col].values, linewidth=0.8)

    axs[0].set_ylabel("Sensor Value")
    axs[0].set_title(f"Sensor Signals – Experiment {exp_id}")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    axs[0].margins(x=0.01)

    categories, starts, ends, colors = [], [], [], []

    for i, label in enumerate(labels):
        mask_true = np.array([1 if y == label else 0 for y in y_true_names])
        mask_pred = np.array([1 if y == label else 0 for y in y_pred_names])

        base_color = base_colors[i]
        true_color = mcolors.to_rgba(base_color, alpha=0.7)
        pred_color = mcolors.to_rgba(base_color, alpha=0.3)

        def extract_segments(mask, category, color):
            start = None
            for j, val in enumerate(mask):
                if val == 1 and start is None:
                    start = j
                elif val == 0 and start is not None:
                    categories.append(category)
                    starts.append(start)
                    ends.append(j - 1)  
                    colors.append(color)
                    start = None
            if start is not None:
                categories.append(category)
                starts.append(start)
                ends.append(len(mask) - 1)  
                colors.append(color)

        extract_segments(mask_true, f"{label} True", true_color)
        extract_segments(mask_pred, f"{label} Pred", pred_color)

    if categories:
        axs[1].barh(
            categories,
            [e - s for s, e in zip(starts, ends)],
            left=starts,
            color=colors,
            height=0.6,
            edgecolor='none'  
        )
    else:
        axs[1].text(
            0.5,
            0.5,
            "No Segments Found",
            transform=axs[1].transAxes,
            ha="center",
            va="center",
        )

    axs[1].set_xlabel("Time Index")
    axs[1].margins(x=0.01)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

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
            
            exp_data = sensor_df[sensor_df["Experiment_ID"] == str(exp_id)]
            if exp_data.empty:
                logger.warning(f"No data found for Experiment ID {exp_id}")
                continue
            
            y_pred_names, y_true_names, timestamps = _predict_experiment(
                model, exp_data, feature_cols, idx_to_label, device
            )
            
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