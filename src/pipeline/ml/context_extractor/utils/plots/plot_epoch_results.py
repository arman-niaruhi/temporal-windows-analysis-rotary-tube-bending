"""
Visualization of training-time (epoch-level) model behavior.

This module generates plots that illustrate model predictions, attention
distributions, and selected input features during training.
These figures are used to analyze learning dynamics and convergence behavior.

All plots in this module are generated during training or validation epochs
and are distinct from post-training interpretability analyses.
"""
# plots/plot_epoch_results.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import random
import torch.nn as nn

from src.pipeline.ml.context_extractor.utils.plots.plot_attention import (
    plot_selected_features_with_attn_heatmap,
)


def plot_tcn_feature_maps(
    tcn_activations: torch.Tensor,
    save_path: Path,
    title: str,
    channels: list[int] | None = None,
    sample_index: int = 0,
) -> None:
    if tcn_activations is None:
        return
    if tcn_activations.dim() != 3:
        return
    tcn_np = tcn_activations.detach().cpu().numpy()
    b, c, t = tcn_np.shape
    if b == 0 or c == 0 or t == 0:
        return

    sample_index = max(0, min(sample_index, b - 1))
    channel_indices = channels if channels is not None else list(range(c))
    channel_indices = [idx for idx in channel_indices if 0 <= idx < c]
    if not channel_indices:
        return

    n_channels = len(channel_indices)
    ncols = min(4, n_channels)
    nrows = int(np.ceil(n_channels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.5 * ncols, 2.8 * nrows),
        squeeze=False,
    )

    x_axis = np.arange(t)
    for i, ch in enumerate(channel_indices):
        row = i // ncols
        col = i % ncols
        ax = axes[row][col]
        ax.plot(x_axis, tcn_np[sample_index, ch, :], lw=1.6, color="#1f77b4")
        ax.set_title(f"Channel {ch}", fontsize=10, weight="bold")
        ax.grid(alpha=0.3, linestyle=":")
        if row == nrows - 1:
            ax.set_xlabel("Time", fontsize=9)
        if col == 0:
            ax.set_ylabel("Activation", fontsize=9)

    for idx in range(n_channels, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    fig.suptitle(title, fontsize=14, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_validation_scatter(
    val_targets: torch.Tensor,
    val_preds: torch.Tensor,
    target_feature_names: list,
    val_pred_dir: Path,
    epoch: int,
) -> None:
    val_targets_np = val_targets.detach().cpu().numpy()
    val_preds_np = val_preds.detach().cpu().numpy()
    n_features = val_targets_np.shape[-1]
    ncols = min(3, n_features)
    nrows = int(np.ceil(n_features / ncols))

    fig_scatter, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    for feat in range(n_features):
        row = feat // ncols
        col = feat % ncols
        ax = axes[row][col]
        true_flat = val_targets_np[..., feat].reshape(-1)
        pred_flat = val_preds_np[..., feat].reshape(-1)
        ax.scatter(true_flat, pred_flat, s=8, alpha=0.35, color="#1f77b4")
        min_val = float(np.min([true_flat.min(), pred_flat.min()]))
        max_val = float(np.max([true_flat.max(), pred_flat.max()]))
        ax.plot([min_val, max_val], [min_val, max_val], color="black", lw=1)
        title = target_feature_names[feat] if target_feature_names else f"feature_{feat}"
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("True")
        ax.set_ylabel("Prediction")
        ax.grid(alpha=0.25)

    for idx in range(n_features, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    fig_scatter.suptitle(
        f"Validation True vs Prediction – Epoch {epoch}",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    val_pred_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = val_pred_dir / f"val_scatter_epoch_{epoch:04d}.png"
    fig_scatter.savefig(scatter_path, dpi=160, bbox_inches="tight")
    plt.close(fig_scatter)

def generate_epoch_plots(
    model: nn.Module,
    plot_X: torch.Tensor,
    plot_Y: torch.Tensor,
    springback: torch.Tensor,
    experiment_config: torch.Tensor,
    X_train: torch.Tensor,
    sensor_names: list,
    target_feature_names: list,
    val_losses: list,
    process_part: str,
    train_losses: list,
    epoch: int,
    y_lim,
    predictions_out: int,
    predictions_dir: Path,
    attention_csv_dir: Path,
    attention_dir: Path,
    annot_timesteps: list,
    mandrel_extraction_annot_timesteps: list,
    n_samples: int = 4,
    target_scaler=None,
) -> None:
    """Save each subplot as a separate image in organized folders
    
    Args:
        sensor_data: Array of shape (n_samples, timesteps, n_features)
        feature_names: List of sensor feature names
        output_feature_names: List of target feature names
        pred_data: Tuple (true_np, pred_np, idxs) for predictions
        loss_data: Tuple (epochs_list, val_losses, train_losses) for loss plot
        attn_data: Attention weights of shape (n_prediction_heads, timesteps)
        epoch: Current epoch number
        x_axis: X-axis values for prediction plots
        y_lim: Y-axis limits for prediction plots
        PREDICTIONS_OUT: Total number of predictions
        train_loss: Training loss value for current epoch
        val_loss: Validation loss value for current epoch
        best_val_loss: Best validation loss so far
        annot_timesteps: List of timesteps to annotate
        mandrel_extraction_annot_timesteps: Optional list of timesteps for mandrel extraction annotation
        
    Returns:
        Paths to saved images: (pred_path, loss_path, attn_path, csv_path)
    """
    
    with torch.no_grad():
        pred, attn = model(plot_X, springback, experiment_config)
        pred_np = pred.cpu().numpy()
        true_np = plot_Y.cpu().numpy()
        attn_mean = attn.mean(0).cpu().numpy() if attn is not None else None

    if target_scaler is not None:
        flat_pred = pred_np.reshape(-1, pred_np.shape[-1])
        flat_true = true_np.reshape(-1, true_np.shape[-1])
        pred_np = target_scaler.inverse_transform(flat_pred).reshape(pred_np.shape)
        true_np = target_scaler.inverse_transform(flat_true).reshape(true_np.shape)

    idxs = random.sample(range(len(true_np)), min(n_samples, len(true_np)))
    x_axis = np.arange(predictions_out)
    
    pred_data = (true_np, pred_np, idxs)
    loss_data = (list(range(1, len(val_losses) + 1)), val_losses, train_losses)
    attn_data = attn_mean
    
    
    sensor_data = X_train
    feature_names = sensor_names

    plt.style.use("tableau-colorblind10")

    true_np, pred_np, idxs = pred_data
    num_samples = len(idxs)
    n_features = true_np.shape[-1]

    nrows = num_samples
    ncols = n_features

    fig_pred, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        sharex=True,
        sharey=False,
    )

    axes = np.array(axes).reshape(nrows, ncols)

    for row_i, idx in enumerate(idxs):
        for feat in range(n_features):
            ax = axes[row_i, feat]

            ax.plot(
                x_axis,
                true_np[idx, :, feat],
                "o-",
                lw=2.2,
                ms=4,
                label="True Value",
            )

            ax.plot(
                x_axis,
                pred_np[idx, :, feat],
                "--s",
                lw=1.8,
                ms=4,
                alpha=0.9,
                label="Prediction",
            )

            if isinstance(y_lim, (list, tuple)) and y_lim and isinstance(y_lim[0], (list, tuple)):
                ax.set_ylim(*y_lim[feat])
            else:
                ax.set_ylim(*y_lim)
            ax.grid(True, linestyle=":", alpha=0.55)

            if feat == 0:
                ax.set_ylabel(f"Sample {row_i}", fontsize=12, weight="bold")

            if row_i == 0:
                ax.set_title(target_feature_names[feat], fontsize=13, weight="bold")

            if feat == n_features - 1:
                ax.legend(fontsize=9, loc="upper right")

    fig_pred.suptitle(
        f"Predictions – Epoch {epoch} ({num_samples} samples × {n_features} features)",
        fontsize=16,
        weight="bold",
    )

    fig_pred.supxlabel(f"Prediction Index (Total: {predictions_out})", fontsize=13)
    fig_pred.supylabel("Target Value", fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    pred_path = predictions_dir / f"predictions_epoch_{epoch:04d}.png"
    fig_pred.savefig(pred_path, dpi=180, bbox_inches="tight")
    plt.close(fig_pred)
    
    if attn_data is not None:
        if attn_data.ndim == 3:
            for feat_idx in range(attn_data.shape[0]):
                attn_mean = attn_data[feat_idx]
                attn_path = attention_dir / f"attention_epoch_{epoch:04d}_feat_{feat_idx:02d}.png"
                plot_selected_features_with_attn_heatmap(
                    process_part,
                    sensor_data,
                    feature_names,
                    attn_mean,
                    attn_path,
                    annot_timesteps,
                    mandrel_extraction_annot_timesteps,
                )
                attn_df = pd.DataFrame(
                    attn_mean,
                    index=[f"Pred_{i}" for i in range(attn_mean.shape[0])],
                    columns=[f"Time_{i}" for i in range(attn_mean.shape[1])],
                )
                csv_path = attention_csv_dir / f"attention_epoch_{epoch:04d}_feat_{feat_idx:02d}.csv"
                attn_df.to_csv(csv_path, float_format="%.6f")

        else:
            attn_mean = attn_data
            attn_path = attention_dir / f"attention_epoch_{epoch:04d}.png"
            plot_selected_features_with_attn_heatmap(
                process_part,
                sensor_data,
                feature_names,
                attn_mean,
                attn_path,
                annot_timesteps,
                mandrel_extraction_annot_timesteps,
            )
            attn_df = pd.DataFrame(
                attn_mean,
                index=[f"Pred_{i}" for i in range(attn_mean.shape[0])],
                columns=[f"Time_{i}" for i in range(attn_mean.shape[1])],
            )
            csv_path = attention_csv_dir / f"attention_epoch_{epoch:04d}.csv"
            attn_df.to_csv(csv_path, float_format="%.6f")
