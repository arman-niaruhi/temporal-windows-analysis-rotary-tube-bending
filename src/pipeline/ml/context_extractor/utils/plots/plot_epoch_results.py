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

from src.pipeline.ml.context_extractor.utils.plots.plot_attention import plot_selected_features_with_attn_heatmap

def generate_epoch_plots(model: nn.Module, plot_X: torch.Tensor, plot_Y: torch.Tensor, springback,
                        X_train: torch.Tensor, sensor_names: list, 
                        target_feature_names: list, val_losses: list, 
                        process_part: str,
                        train_losses: list, epoch: int,
                        y_lim: tuple, predictions_out: int, train_loss: float,
                        val_loss: float, best_val_loss: float, predictions_dir: Path,
                        attention_csv_dir: Path, attention_dir: Path, loss_dir: Path,
                        annot_timesteps: list, mandrel_extraction_annot_timesteps: list,
                        n_samples: int = 4) -> None:
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
        pred, attn = model(plot_X, springback)
        pred_np = pred.cpu().numpy()
        true_np = plot_Y.cpu().numpy()
        attn_mean = attn.mean(0).cpu().numpy()

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

    fig_loss = plt.figure(figsize=(10, 7))
    ax_loss = fig_loss.add_subplot(111)

    epochs_list, val_losses, train_losses = loss_data

    ax_loss.plot(
        epochs_list,
        train_losses,
        color="#1f77b4",
        lw=3,
        alpha=0.7,
        label="Train MSE",
    )
    ax_loss.plot(epochs_list, val_losses, color="#d62728", lw=3, label="Val MSE")
    ax_loss.plot(
        epochs_list,
        [best_val_loss] * len(epochs_list),
        color="green",
        lw=2.5,
        ls="--",
        label="Best Val MSE",
    )

    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("MSE", fontsize=12)
    ax_loss.set_title(
        f"Training Progress - Epoch {epoch}\nTrain: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f}",
        fontweight="bold",
        fontsize=14,
    )
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=10)
    plt.tight_layout()

    loss_path = loss_dir / f"loss_epoch_{epoch:04d}.png"
    fig_loss.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig_loss)
    
    attn_mean = attn_data
    attn_path = attention_dir / f"attention_epoch_{epoch:04d}.png"
    plot_selected_features_with_attn_heatmap(
        process_part, sensor_data, feature_names, attn_mean, attn_path, annot_timesteps, mandrel_extraction_annot_timesteps
    )
    attn_df = pd.DataFrame(
        attn_mean,
        index=[f"Pred_{i}" for i in range(attn_mean.shape[0])],
        columns=[f"Time_{i}" for i in range(attn_mean.shape[1])],
    )

    csv_path = attention_csv_dir / f"attention_epoch_{epoch:04d}.csv"
    attn_df.to_csv(csv_path, float_format="%.6f")