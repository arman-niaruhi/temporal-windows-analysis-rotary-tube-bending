import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
import logging
import matplotlib.pyplot as plt
import gc
from pathlib import Path
import os
from matplotlib import rcParams
import pandas as pd

logger = logging.getLogger(__name__)

def get_plot_batch(plot_loader: DataLoader, device: torch.device) -> tuple:
    """Get a batch of data for plotting."""
    plot_X, plot_Y = next(iter(plot_loader))
    plot_X = plot_X.to(device)
    return plot_X, plot_Y

def compute_plot_limits(Y_val: torch.Tensor) -> tuple:
    """Compute global y-limits for consistent plotting."""
    y_all = Y_val[:, :, 0].cpu().numpy()
    global_ymin, global_ymax = y_all.min(), y_all.max()
    margin = (global_ymax - global_ymin) * 0.1
    return (global_ymin - margin, global_ymax + margin)


def generate_epoch_plots(model: nn.Module, plot_X: torch.Tensor, plot_Y: torch.Tensor,
                        X_train: torch.Tensor, sensor_names: list, 
                        target_feature_names: list, val_losses: list, 
                        machine_part: str,
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
        pred, attn = model(plot_X)
        pred_np = pred.cpu().numpy()
        true_np = plot_Y.cpu().numpy()
        attn_mean = attn.mean(0).cpu().numpy()

    idxs = random.sample(range(len(true_np)), min(n_samples, len(true_np)))
    x_axis = np.arange(predictions_out)
    
    pred_data = (true_np, pred_np, idxs)
    loss_data = (list(range(1, len(val_losses) + 1)), val_losses, train_losses)
    attn_data = attn_mean
    
    
    sensor_data = X_train

    output_feature_names = target_feature_names
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
                ax.set_title(output_feature_names[feat], fontsize=13, weight="bold")

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
        machine_part, sensor_data, feature_names, attn_mean, attn_path, annot_timesteps, mandrel_extraction_annot_timesteps
    )
    attn_df = pd.DataFrame(
        attn_mean,
        index=[f"Pred_{i}" for i in range(attn_mean.shape[0])],
        columns=[f"Time_{i}" for i in range(attn_mean.shape[1])],
    )

    csv_path = attention_csv_dir / f"attention_epoch_{epoch:04d}.csv"
    attn_df.to_csv(csv_path, float_format="%.6f")

    
def generate_final_attention_plot(model: nn.Module, plot_X: torch.Tensor,
                                 X_val: torch.Tensor, sensor_names: list,
                                 machine_part: str,
                                 attention_lines_dir : Path,
                                 annot_timesteps: list,
                                 mandrel_extraction_annot_timesteps: list) -> None:
    """Generate final attention visualization."""
    """
    Plots sensor data and ONE attention head as line plots in two subplots.
    Creates separate plots for each attention head (angle).
    
    Args:
        sensor_data: Array of shape (n_samples, timesteps, n_features)
        sensor_names: List of sensor feature names
        attn_mean: Attention weights of shape (n_prediction_heads, timesteps)
        annot_timesteps: Optional list of timesteps to annotate
        sample_idx: Which sample to plot (default -1 for last sample)
        figsize: Figure size tuple
    """
    with torch.no_grad():
        model.eval()
        _, final_attn = model(plot_X)
        final_attn_mean = final_attn.mean(0).cpu().numpy()

    
    sensor_data=X_val
    attn_mean=final_attn_mean
    sample_idx=-1
    figsize: tuple=(20, 10)
    rcParams["font.family"] = "sans-serif"
    rcParams["font.size"] = 10

    cleaned_feature_names = [name.replace("_mean", "") for name in sensor_names]

    sample_data = sensor_data[sample_idx, :, :]
    main_timesteps = sample_data.shape[0]
    n_attention_heads = attn_mean.shape[0]
    attn_timesteps = attn_mean.shape[1]

    if attn_timesteps != main_timesteps:
        attn_data_resized = np.zeros((n_attention_heads, main_timesteps))
        for i in range(n_attention_heads):
            x_original = np.arange(attn_timesteps)
            x_target = np.linspace(0, attn_timesteps - 1, main_timesteps)
            attn_data_resized[i] = np.interp(x_target, x_original, attn_mean[i])
        attn_data = attn_data_resized
    else:
        attn_data = attn_mean

    time_steps = np.arange(main_timesteps)
    cmap = plt.get_cmap("tab20")
    colors_sensors = cmap(np.linspace(0, 1, len(cleaned_feature_names)))
    
    annot_labels = None
    if annot_timesteps and (machine_part == "All"):
        annot_labels = [
            "Start-Clamping",
            "Start-Bending",
            "Start-Declamping",
            "End-Declamping",
        ]

    saved_paths = []

    for angle_idx in range(n_attention_heads):
        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.clf()
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
        ax_sensors = fig.add_subplot(gs[0])
        ax_attention = fig.add_subplot(gs[1])

        for i, (feature_name, color) in enumerate(zip(cleaned_feature_names, colors_sensors)):
            ax_sensors.plot(
                time_steps,
                sample_data[:, i],
                color=color,
                linewidth=2.0,
                alpha=0.85,
                label=feature_name,
                marker="o",
                markersize=3,
                markevery=max(1, main_timesteps // 20),
            )

        ax_sensors.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
        ax_sensors.set_ylabel("Sensor Value", fontsize=12, fontweight="bold", labelpad=10)
        ax_sensors.grid(True, alpha=0.2, linestyle="--", linewidth=0.8, color="gray")
        ax_sensors.set_axisbelow(True)

        ax_sensors.spines["top"].set_visible(False)
        ax_sensors.spines["right"].set_visible(False)
        ax_sensors.spines["left"].set_linewidth(1.2)
        ax_sensors.spines["bottom"].set_linewidth(1.2)
        ax_sensors.spines["left"].set_color("#333333")
        ax_sensors.spines["bottom"].set_color("#333333")

        if annot_labels:
            for ts, label in zip(annot_timesteps, annot_labels):
                ax_sensors.axvline(
                    ts, color="black", linestyle="--", linewidth=1.2, alpha=0.6
                )
                ax_sensors.annotate(
                    label,
                    xy=(ts, sample_data[:, :].max()),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8),
                )
        if mandrel_extraction_annot_timesteps and (machine_part == "All"):
            ax_sensors.axvspan(
                mandrel_extraction_annot_timesteps[0],
                mandrel_extraction_annot_timesteps[1],
                color="blue",
                alpha=0.12,
                linewidth=0,
                zorder=0.5
            )
            
        ax_sensors.set_xlim(0, main_timesteps - 1)
        ax_sensors.set_facecolor("#f9f9f9")
        ax_sensors.set_title(
            "Sensor Data Over Time",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        legend_sensors = ax_sensors.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=9,
            framealpha=0.95,
            edgecolor="#cccccc",
            ncol=1,
        )
        legend_sensors.get_frame().set_facecolor("white")
        legend_sensors.get_frame().set_linewidth(1.2)

        display_angle = angle_idx + 1
        
        # Use red color for all attention plots
        angle_color = '#d62728'  # Red color
        
        ax_attention.plot(
            time_steps,
            attn_data[angle_idx, :],
            color=angle_color,
            linewidth=2.5,
            alpha=0.9,
            label=f"Angle {display_angle}",
            marker="s",
            markersize=4,
            markevery=max(1, main_timesteps // 20),
        )

        ax_attention.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
        ax_attention.set_ylabel("Attention Weight", fontsize=12, fontweight="bold", labelpad=10)
        ax_attention.grid(True, alpha=0.2, linestyle="--", linewidth=0.8, color="gray")
        ax_attention.set_axisbelow(True)

        ax_attention.spines["top"].set_visible(False)
        ax_attention.spines["right"].set_visible(False)
        ax_attention.spines["left"].set_linewidth(1.2)
        ax_attention.spines["bottom"].set_linewidth(1.2)
        ax_attention.spines["left"].set_color("#333333")
        ax_attention.spines["bottom"].set_color("#333333")

        if annot_labels:
            for ts, label in zip(annot_timesteps, annot_labels):
                ax_attention.axvline(
                    ts, color="black", linestyle="--", linewidth=1.2, alpha=0.6
                )
                ax_attention.annotate(
                    label,
                    xy=(ts, attn_data[angle_idx, :].max()),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8),
                )
        if mandrel_extraction_annot_timesteps and (machine_part == "All"):
            ax_attention.axvspan(
                mandrel_extraction_annot_timesteps[0],
                mandrel_extraction_annot_timesteps[1],
                color="blue",
                alpha=0.12,
                linewidth=0,
                zorder=0.5
            )
        ax_attention.set_xlim(0, main_timesteps - 1)
        ax_attention.set_facecolor("#f9f9f9")
        ax_attention.set_title(
            f"Attention Weight - Angle {display_angle}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        legend_attention = ax_attention.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=11,
            framealpha=0.95,
            edgecolor="#cccccc",
        )
        legend_attention.get_frame().set_facecolor("white")
        legend_attention.get_frame().set_linewidth(1.2)

        fig.suptitle(
            f"Final Epoch - Angle {display_angle} Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout()

        line_path = attention_lines_dir / f"attention_angle_{display_angle:02d}.png"
        fig.savefig(line_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        saved_paths.append(line_path)
        
        

def plot_attention_lines_with_sensors(
    sensor_data: np.ndarray,
    sensor_names: list,
    attn_mean: np.ndarray,
    machine_part: str,
    attention_lines_dir: Path,
    annot_timesteps: list = None,
    mandrel_extraction_annot_timesteps: list = None,
    sample_idx: int = -1,
    figsize: tuple=(20, 10),
):
    """
    Plots sensor data and ONE attention head as line plots in two subplots.
    Creates separate plots for each attention head (angle).
    
    Args:
        sensor_data: Array of shape (n_samples, timesteps, n_features)
        sensor_names: List of sensor feature names
        attn_mean: Attention weights of shape (n_prediction_heads, timesteps)
        annot_timesteps: Optional list of timesteps to annotate
        sample_idx: Which sample to plot (default -1 for last sample)
        figsize: Figure size tuple
    """
    rcParams["font.family"] = "sans-serif"
    rcParams["font.size"] = 10

    cleaned_feature_names = [name.replace("_mean", "") for name in sensor_names]

    sample_data = sensor_data[sample_idx, :, :]
    main_timesteps = sample_data.shape[0]
    n_attention_heads = attn_mean.shape[0]
    attn_timesteps = attn_mean.shape[1]

    if attn_timesteps != main_timesteps:
        attn_data_resized = np.zeros((n_attention_heads, main_timesteps))
        for i in range(n_attention_heads):
            x_original = np.arange(attn_timesteps)
            x_target = np.linspace(0, attn_timesteps - 1, main_timesteps)
            attn_data_resized[i] = np.interp(x_target, x_original, attn_mean[i])
        attn_data = attn_data_resized
    else:
        attn_data = attn_mean

    time_steps = np.arange(main_timesteps)
    cmap = plt.get_cmap("tab20")
    colors_sensors = cmap(np.linspace(0, 1, len(cleaned_feature_names)))
    
    annot_labels = None
    if annot_timesteps and (machine_part == "All"):
        annot_labels = [
            "Start-Clamping",
            "Start-Bending",
            "Start-Declamping",
            "End-Declamping",
        ]

    saved_paths = []

    for angle_idx in range(n_attention_heads):
        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.clf()
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
        ax_sensors = fig.add_subplot(gs[0])
        ax_attention = fig.add_subplot(gs[1])

        for i, (feature_name, color) in enumerate(zip(cleaned_feature_names, colors_sensors)):
            ax_sensors.plot(
                time_steps,
                sample_data[:, i],
                color=color,
                linewidth=2.0,
                alpha=0.85,
                label=feature_name,
                marker="o",
                markersize=3,
                markevery=max(1, main_timesteps // 20),
            )

        ax_sensors.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
        ax_sensors.set_ylabel("Sensor Value", fontsize=12, fontweight="bold", labelpad=10)
        ax_sensors.grid(True, alpha=0.2, linestyle="--", linewidth=0.8, color="gray")
        ax_sensors.set_axisbelow(True)

        ax_sensors.spines["top"].set_visible(False)
        ax_sensors.spines["right"].set_visible(False)
        ax_sensors.spines["left"].set_linewidth(1.2)
        ax_sensors.spines["bottom"].set_linewidth(1.2)
        ax_sensors.spines["left"].set_color("#333333")
        ax_sensors.spines["bottom"].set_color("#333333")

        if annot_labels:
            for ts, label in zip(annot_timesteps, annot_labels):
                ax_sensors.axvline(
                    ts, color="black", linestyle="--", linewidth=1.2, alpha=0.6
                )
                ax_sensors.annotate(
                    label,
                    xy=(ts, sample_data[:, :].max()),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8),
                )
        if mandrel_extraction_annot_timesteps and (machine_part == "All"):
            ax_sensors.axvspan(
                mandrel_extraction_annot_timesteps[0],
                mandrel_extraction_annot_timesteps[1],
                color="blue",
                alpha=0.12,
                linewidth=0,
                zorder=0.5
            )
            
        ax_sensors.set_xlim(0, main_timesteps - 1)
        ax_sensors.set_facecolor("#f9f9f9")
        ax_sensors.set_title(
            "Sensor Data Over Time",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        legend_sensors = ax_sensors.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=9,
            framealpha=0.95,
            edgecolor="#cccccc",
            ncol=1,
        )
        legend_sensors.get_frame().set_facecolor("white")
        legend_sensors.get_frame().set_linewidth(1.2)

        display_angle = angle_idx + 1
        
        # Use red color for all attention plots
        angle_color = '#d62728'  # Red color
        
        ax_attention.plot(
            time_steps,
            attn_data[angle_idx, :],
            color=angle_color,
            linewidth=2.5,
            alpha=0.9,
            label=f"Angle {display_angle}",
            marker="s",
            markersize=4,
            markevery=max(1, main_timesteps // 20),
        )

        ax_attention.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
        ax_attention.set_ylabel("Attention Weight", fontsize=12, fontweight="bold", labelpad=10)
        ax_attention.grid(True, alpha=0.2, linestyle="--", linewidth=0.8, color="gray")
        ax_attention.set_axisbelow(True)

        ax_attention.spines["top"].set_visible(False)
        ax_attention.spines["right"].set_visible(False)
        ax_attention.spines["left"].set_linewidth(1.2)
        ax_attention.spines["bottom"].set_linewidth(1.2)
        ax_attention.spines["left"].set_color("#333333")
        ax_attention.spines["bottom"].set_color("#333333")

        if annot_labels:
            for ts, label in zip(annot_timesteps, annot_labels):
                ax_attention.axvline(
                    ts, color="black", linestyle="--", linewidth=1.2, alpha=0.6
                )
                ax_attention.annotate(
                    label,
                    xy=(ts, attn_data[angle_idx, :].max()),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8),
                )
        if mandrel_extraction_annot_timesteps and (machine_part == "All"):
            ax_attention.axvspan(
                mandrel_extraction_annot_timesteps[0],
                mandrel_extraction_annot_timesteps[1],
                color="blue",
                alpha=0.12,
                linewidth=0,
                zorder=0.5
            )
        ax_attention.set_xlim(0, main_timesteps - 1)
        ax_attention.set_facecolor("#f9f9f9")
        ax_attention.set_title(
            f"Attention Weight - Angle {display_angle}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        legend_attention = ax_attention.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=11,
            framealpha=0.95,
            edgecolor="#cccccc",
        )
        legend_attention.get_frame().set_facecolor("white")
        legend_attention.get_frame().set_linewidth(1.2)

        fig.suptitle(
            f"Final Epoch - Angle {display_angle} Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout()

        line_path = attention_lines_dir / f"attention_angle_{display_angle:02d}.png"
        fig.savefig(line_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        saved_paths.append(line_path)

    return saved_paths

def plot_selected_features_with_attn_heatmap(
    machine_part,
    sensor_data: np.ndarray,
    sensor_names: list,
    attn_mean: np.ndarray,
    attn_path: Path,
    annot_timesteps: list = None,
    mandrel_extraction_annot_timesteps: list = None,
    figsize: tuple = (25, 12),
):
    """
    Plots selected features with attention heatmap at the bottom.
    Includes legend for the top plot on the right side.
    Enhanced with beautiful styling and improved aesthetics.
    
    Args:
        sensor_data: Array of shape (n_samples, timesteps, n_features)
        sensor_names: List of sensor feature names
        attn_mean: Attention weights of shape (n_prediction_heads, timesteps)
        attn_path: Path to save the attention plot
        annot_timesteps: Optional list of timesteps to annotate
        mandrel_extraction_annot_timesteps: Optional list of timesteps for mandrel extraction annotation
        sample_idx: Which sample to plot (default last sample)
        figsize: Figure size tuple
    """
    rcParams["font.family"] = "sans-serif"
    rcParams["font.size"] = 10

    cleaned_feature_names = [name.replace("_mean", "") for name in sensor_names]

    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.clf()
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.25, wspace=0.3)
    ax_main = fig.add_subplot(gs[0])
    ax_heatmap = fig.add_subplot(gs[1])

    sample_data = sensor_data[-1, :, :]
    main_timesteps = sample_data.shape[0]
    n_attention_heads = attn_mean.shape[0]
    attn_timesteps = attn_mean.shape[1]

    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(cleaned_feature_names)))

    for i, (feature_name, color) in enumerate(zip(cleaned_feature_names, colors)):
        ax_main.plot(
            sample_data[:, i],
            color=color,
            linewidth=2.5,
            alpha=0.85,
            label=feature_name,
            marker="o",
            markersize=3,
            markevery=max(1, main_timesteps // 20),
        )  

    ax_main.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
    ax_main.set_ylabel("Feature Value", fontsize=12, fontweight="bold", labelpad=10)
    ax_main.grid(True, alpha=0.2, linestyle="--", linewidth=0.8, color="gray")
    ax_main.set_axisbelow(True)

    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["left"].set_linewidth(1.2)
    ax_main.spines["bottom"].set_linewidth(1.2)
    ax_main.spines["left"].set_color("#333333")
    ax_main.spines["bottom"].set_color("#333333")

    if annot_timesteps and (machine_part == "All"):
        annot_labels = [
            "Start-Declamping",
            "Start-Bending",
            "Start-Declamping",
            "End-Declamping",
        ]  

    
        for ts, label in zip(annot_timesteps, annot_labels):
            ax_main.axvline(
                ts, color="black", linestyle="--", linewidth=1.2, alpha=0.7
            )

            ax_main.annotate(
                label,
                xy=(ts, sample_data[:, :].max()),  # anchor at top of plot
                xytext=(0, 10),  # offset upward
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8),
            )

    if mandrel_extraction_annot_timesteps and (machine_part == "All"):
        ax_main.axvspan(
            mandrel_extraction_annot_timesteps[0],
            mandrel_extraction_annot_timesteps[1],
            color="blue",
            alpha=0.12,
            linewidth=0,
            zorder=0.5
        )
    ax_main.set_xlim(0, main_timesteps - 1)
    ax_main.set_facecolor("#f9f9f9")
    ax_main.set_title(
        "Sensor Data Over Time", fontsize=14, fontweight="bold", pad=15
    )

    legend = ax_main.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        framealpha=0.95,
        edgecolor="#cccccc",
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_linewidth(1.2)

    if attn_timesteps != main_timesteps:
        attn_data_resized = np.zeros((n_attention_heads, main_timesteps))
        for i in range(n_attention_heads):
            x_original = np.arange(attn_timesteps)
            x_target = np.linspace(0, attn_timesteps - 1, main_timesteps)
            attn_data_resized[i] = np.interp(x_target, x_original, attn_mean[i])
        attn_data = attn_data_resized
    else:
        attn_data = attn_mean

    im = ax_heatmap.imshow(
        attn_data,
        aspect="auto",
        cmap="magma",  
        interpolation="bilinear",
        extent=[0, main_timesteps - 1, 0, n_attention_heads - 1],
    )

    ax_heatmap.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
    ax_heatmap.set_ylabel(
        "Attention Head", fontsize=9, fontweight="bold", labelpad=10
    )

    ax_heatmap.set_yticks(np.arange(n_attention_heads))
    ax_heatmap.set_yticklabels(
        [f"{i + 1}" for i in reversed(range(n_attention_heads))], fontsize=5
    )

    ax_heatmap.set_xlim(0, main_timesteps - 1)
    ax_heatmap.set_facecolor("white")
    ax_heatmap.set_title(
        "Attention Head Intensity", fontsize=14, fontweight="bold", pad=15
    )

    ax_heatmap.spines["top"].set_visible(False)
    ax_heatmap.spines["right"].set_visible(False)
    ax_heatmap.spines["left"].set_linewidth(1.2)
    ax_heatmap.spines["bottom"].set_linewidth(1.2)
    ax_heatmap.spines["left"].set_color("#333333")
    ax_heatmap.spines["bottom"].set_color("#333333")

    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.9, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=11, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(1.2)

    plt.tight_layout()

    pos_main = ax_main.get_position()
    pos_heat = ax_heatmap.get_position()

    ax_heatmap.set_position(
        [pos_heat.x0, pos_heat.y0, pos_main.width, pos_heat.height]
    )

    fig.savefig(attn_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    cbar.ax.set_position(
        [pos_main.x0 + pos_main.width + 0.02, pos_heat.y0, 0.015, pos_heat.height]
    )

    fig.savefig(attn_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    

    

def __create_metrics_summary_file(metrics_history: dict, train_losses: list,
                               val_losses: list, epoch_times: list,
                               learning_rates: list, saving_dir) -> Path:
    """Create a text file summarizing all training metrics."""
    epochs = list(range(1, len(train_losses) + 1))
    summary_path = saving_dir / "08_metrics" / "metrics_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FINAL TRAINING METRICS SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Epochs:          {len(epochs)}\n")
        f.write(f"Best Val Loss:         {min(val_losses):.6f}\n")
        f.write(f"Final Val Loss:        {val_losses[-1]:.6f}\n")
        f.write(f"Final Train Loss:      {train_losses[-1]:.6f}\n\n")
        f.write("-"*60 + "\n")
        f.write("FINAL VALIDATION METRICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"MSE:                   {metrics_history['mse'][-1]:.6f}\n")
        f.write(f"RMSE:                  {metrics_history['rmse'][-1]:.6f}\n")
        f.write(f"MAE:                   {metrics_history['mae'][-1]:.6f}\n")
        f.write(f"MedAE:                 {metrics_history['medae'][-1]:.6f}\n")
        f.write(f"R² Score:              {metrics_history['r2'][-1]:.6f}\n")
        f.write(f"MAPE:                  {metrics_history['mape'][-1]:.2f}%\n")
        f.write(f"Max Error:             {metrics_history['max_error'][-1]:.6f}\n")
        f.write(f"EVS:                   {metrics_history['evs'][-1]:.6f}\n")
        f.write(f"MBE:                   {metrics_history['mbe'][-1]:.6f}\n\n")
        f.write("-"*60 + "\n")
        f.write("TRAINING STATISTICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"Avg Epoch Time:        {np.mean(epoch_times):.2f}s\n")
        f.write(f"Total Training Time:   {sum(epoch_times):.2f}s\n")
        f.write(f"Final Learning Rate:   {learning_rates[-1]:.2e}\n")
        f.write("="*60 + "\n")
    
    logger.info(f"Saved Metrics Summary to {summary_path}")
    return summary_path


def __create_metric_plot(epochs: list, values: list, ylabel: str, title: str,
                      color: str, output_path: Path, reference_line: dict = None) -> None:
    """Create and save a single metric plot."""
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, values, color=color, linewidth=2.5, marker='o', markersize=4)
    
    if reference_line:
        ax.axhline(y=reference_line['y'], color=reference_line.get('color', 'gray'),
                  linestyle='--', alpha=0.5, linewidth=2,
                  label=reference_line.get('label', ''))
        ax.legend(fontsize=12)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close("all")
    gc.collect()
    logger.info(f"Saved {title} plot to {output_path}")


def __plot_loss_curves(epochs: list, train_losses: list, val_losses: list, saving_dir: Path) -> Path:
    """Create and save training and validation loss curves."""
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label='Train Loss', color='blue',
           linewidth=2.5, marker='o', markersize=4)
    ax.plot(epochs, val_losses, label='Val Loss', color='red',
           linewidth=2.5, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training and Validation Loss', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    metric_path = saving_dir / "08_metrics"
    # Create the directory if it does not exist
    metric_path.mkdir(parents=True, exist_ok=True)
    path = metric_path / "metric_loss.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close("all")
    gc.collect()
    logger.info(f"Saved Loss plot to {path}")
    return path

def plot_all_metrics(metrics_history: dict, train_losses: list[float],
                    val_losses: list[float], learning_rates: list[float],
                    epoch_times: list[float], saving_dir) -> None:
    """Create individual plots for each training metric."""
    epochs = list(range(1, len(train_losses) + 1))
    saved_paths = []
    
    path = __plot_loss_curves(epochs, train_losses, val_losses, saving_dir)
    saved_paths.append(path)
    
    metric_path = saving_dir / "08_metrics"
    
    path = metric_path / "metric_mse.png"
    __create_metric_plot(epochs, metrics_history['mse'], 'MSE',
                      'Mean Squared Error', 'purple', path)
    saved_paths.append(path)
    
    path = metric_path / "metric_rmse.png"
    __create_metric_plot(epochs, metrics_history['rmse'], 'RMSE',
                      'Root Mean Squared Error', 'darkviolet', path)
    saved_paths.append(path)
    
    path = metric_path / "metric_mae.png"
    __create_metric_plot(epochs, metrics_history['mae'], 'MAE',
                      'Mean Absolute Error', 'orange', path)
    saved_paths.append(path)
    
    path = metric_path / "metric_medae.png"
    __create_metric_plot(epochs, metrics_history['medae'], 'MedAE',
                      'Median Absolute Error', 'darkorange', path)
    saved_paths.append(path)
    
    path = metric_path / "metric_r2.png"
    __create_metric_plot(epochs, metrics_history['r2'], 'R² Score',
                      'R² Score (Coefficient of Determination)', 'green', path,
                      reference_line={'y': 1.0, 'label': 'Perfect Score'})
    saved_paths.append(path)
    
    path = metric_path / "metric_mape.png"
    __create_metric_plot(epochs, metrics_history['mape'], 'MAPE (%)',
                      'Mean Absolute Percentage Error', 'brown', path)
    saved_paths.append(path)
    
    path = metric_path / "metric_max_error.png"
    __create_metric_plot(epochs, metrics_history['max_error'], 'Max Error',
                      'Maximum Error', 'red', path)
    saved_paths.append(path)
    
    path = metric_path / "metric_evs.png"
    __create_metric_plot(epochs, metrics_history['evs'], 'EVS',
                      'Explained Variance Score', 'teal', path,
                      reference_line={'y': 1.0, 'label': 'Perfect Score'})
    saved_paths.append(path)
    
    path = metric_path / "metric_mbe.png"
    __create_metric_plot(epochs, metrics_history['mbe'], 'MBE',
                      'Mean Bias Error', 'navy', path,
                      reference_line={'y': 0, 'label': 'Zero Bias'})
    saved_paths.append(path)
    
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, learning_rates, color='magenta', linewidth=2.5,
           marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = metric_path / "metric_learning_rate.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close("all")
    gc.collect()
    saved_paths.append(path)
    logger.info(f"Saved Learning Rate plot to {path}")
    
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, epoch_times, color='cyan', linewidth=2.5,
           marker='o', markersize=4)
    avg_time = np.mean(epoch_times)
    ax.axhline(y=avg_time, color='red', linestyle='--', alpha=0.5,
              linewidth=2, label=f'Avg: {avg_time:.2f}s')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_title('Training Time per Epoch', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = metric_path / "metric_epoch_time.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close("all")
    gc.collect()
    saved_paths.append(path)
    logger.info(f"Saved Epoch Time plot to {path}")
    
    summary_path = __create_metrics_summary_file(
        metrics_history, train_losses, val_losses, epoch_times,
        learning_rates, saving_dir
    )
    saved_paths.append(summary_path)
    logger.info(f"Total of {len(saved_paths)} metric files saved and logged to MLflow")

def plot_feature_importance_bar(importance_df, feature_names, output_path):
    """Plot feature importance as horizontal bar chart."""
    plt.figure(figsize=(14, 10))
    top_features = min(20, len(feature_names))
    top_df = importance_df.head(top_features)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_df)))
    plt.barh(range(len(top_df)), top_df["Attention_Importance"], color=colors, alpha=0.8)

    plt.yticks(range(len(top_df)), top_df["Feature"], fontsize=11)
    plt.xlabel("Attention Importance Score", fontsize=13, fontweight="bold")
    plt.ylabel("Features", fontsize=13, fontweight="bold")
    plt.title(
        f"Top {top_features} Most Important Features\nLSTM Attention-Based Feature Importance",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    plt.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_attention_time_distribution(timestep_attention, output_path):
    """Plot attention distribution over time steps."""
    if len(timestep_attention) <= 1:
        return
        
    plt.figure(figsize=(14, 8))
    x_vals = np.arange(len(timestep_attention))

    plt.plot(
        x_vals,
        timestep_attention,
        "o-",
        color="#2E86AB",
        linewidth=3,
        markersize=8,
        markerfacecolor="#A23B72",
        markeredgecolor="#6D214F",
        markeredgewidth=2,
        label="Attention Weights",
    )

    peak_threshold = np.percentile(timestep_attention, 75)
    peaks = np.where(timestep_attention > peak_threshold)[0]
    plt.scatter(
        peaks,
        timestep_attention[peaks],
        color="#F18F01",
        s=150,
        zorder=5,
        label="High Attention Peaks",
        edgecolors="#C73E1D",
        linewidth=2,
    )

    if len(timestep_attention) > 3:
        z = np.polyfit(x_vals, timestep_attention, 2)
        p = np.poly1d(z)
        plt.plot(
            x_vals,
            p(x_vals),
            "--",
            color="#C73E1D",
            linewidth=2,
            alpha=0.7,
            label="Trend Line",
        )

    plt.xlabel("Time Step", fontsize=13, fontweight="bold")
    plt.ylabel("Attention Weight", fontsize=13, fontweight="bold")
    plt.title(
        "Attention Distribution Across Time Steps\nIdentifying Critical Moments in the Sequence",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_attention_heatmap(timestep_attention, feature_names, output_path):
    """Plot attention heatmap across time steps and features."""
    if len(timestep_attention) <= 1:
        return
        
    plt.figure(figsize=(16, 6))

    heatmap_data = np.tile(timestep_attention, (min(15, len(feature_names)), 1))

    im = plt.imshow(
        heatmap_data,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
        extent=[0, len(timestep_attention) - 1, 0, min(15, len(feature_names)) - 1],
    )

    plt.xlabel("Time Steps", fontsize=13, fontweight="bold")
    plt.ylabel("Feature Representation", fontsize=13, fontweight="bold")
    plt.title(
        "Attention Pattern Heatmap\nVisualizing Attention Intensity Across Features and Time",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    cbar = plt.colorbar(im, shrink=0.8, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    feature_indices = np.linspace(0, min(15, len(feature_names)) - 1, 5, dtype=int)
    feature_labels = [
        feature_names[i] if i < len(feature_names) else f"Feature {i}"
        for i in feature_indices
    ]
    plt.yticks(feature_indices, feature_labels, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_cumulative_attention(timestep_attention, output_path):
    """Plot cumulative attention distribution over time."""
    if len(timestep_attention) <= 1:
        return
        
    plt.figure(figsize=(14, 8))

    cumulative_attention = np.cumsum(timestep_attention) / np.sum(timestep_attention)
    plt.plot(
        np.arange(len(timestep_attention)),
        cumulative_attention,
        "g-",
        linewidth=4,
        alpha=0.8,
        label="Cumulative Attention",
    )

    threshold_colors = ["red", "orange", "purple"]
    thresholds = [0.5, 0.8, 0.95]

    for i, (threshold, color) in enumerate(zip(thresholds, threshold_colors)):
        idx = np.where(cumulative_attention >= threshold)[0]
        if len(idx) > 0:
            step = idx[0]
            plt.axvline(
                x=step,
                color=color,
                linestyle="--",
                alpha=0.8,
                linewidth=2,
                label=f"{threshold * 100:.0f}% attention by step {step}",
            )

            plt.annotate(
                f"{threshold * 100:.0f}%",
                xy=(step, threshold),
                xytext=(10, 20 + i * 30),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.7),
            )

    plt.xlabel("Time Step", fontsize=13, fontweight="bold")
    plt.ylabel("Cumulative Attention Fraction", fontsize=13, fontweight="bold")
    plt.title(
        "Cumulative Attention Distribution\nHow Quickly the Model Focuses Its Attention",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    plt.legend(fontsize=11, loc="lower right")
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_importance_rank_distribution(importance_df, output_path):
    """Plot feature importance distribution by rank."""
    plt.figure(figsize=(14, 8))

    ranks = np.arange(1, len(importance_df) + 1)
    importance_values = importance_df["Attention_Importance"].values

    plt.semilogy(
        ranks,
        importance_values,
        "s-",
        color="#6A0572",
        linewidth=2,
        markersize=6,
        alpha=0.8,
        label="Feature Importance",
    )

    top_n = min(10, len(importance_df))
    plt.scatter(
        ranks[:top_n],
        importance_values[:top_n],
        color="#FF6B6B",
        s=100,
        zorder=5,
        label=f"Top {top_n} Features",
        edgecolors="darkred",
        linewidth=2,
    )

    plt.xlabel("Feature Rank", fontsize=13, fontweight="bold")
    plt.ylabel("Attention Importance (Log Scale)", fontsize=13, fontweight="bold")
    plt.title(
        "Feature Importance Distribution by Rank\nIdentifying the Most Influential Features",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, which="both")

    for percentile in [25, 50, 75]:
        value = np.percentile(importance_values, percentile)
        plt.axhline(
            y=value,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label=f"{percentile}th percentile: {value:.4f}",
        )

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_all_plots(importance_df, timestep_attention, mse, feature_names, output_dir):
    """Generate all visualization plots."""
    plot_paths = {}
    
    plot_paths["feature_importance"] = output_dir / "01_feature_importance_bars.png"
    plot_feature_importance_bar(importance_df, feature_names, plot_paths["feature_importance"])
    
    plot_paths["time_distribution"] = output_dir / "02_attention_time_distribution.png"
    plot_attention_time_distribution(timestep_attention, plot_paths["time_distribution"])
    
    plot_paths["heatmap"] = output_dir / "03_attention_heatmap.png"
    plot_attention_heatmap(timestep_attention, feature_names, plot_paths["heatmap"])
    
    plot_paths["cumulative"] = output_dir / "04_cumulative_attention.png"
    plot_cumulative_attention(timestep_attention, plot_paths["cumulative"])
    
    plot_paths["rank_distribution"] = output_dir / "05_importance_rank_distribution.png"
    plot_importance_rank_distribution(importance_df, plot_paths["rank_distribution"])
    
    return plot_paths


def visualize_window_importance(
    angle: int,
    feature_names: list[str],
    mean_importance,
    annot_timesteps,
    window_importance_plots_dir,
    mandrel_extraction_annot_timesteps,
    machine_part,
    occluded_window_size: int = 10,
    stride: int = 5):
    """
    Plot mean window importance across all samples.
    Shows which time windows are most important for predictions.
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    
    # Map window indices to actual timesteps (center of each window)
    n_windows = len(mean_importance)
    window_timesteps = [i * stride + occluded_window_size // 2 for i in range(n_windows)]
    
    ax.fill_between(
        window_timesteps,
        0,
        mean_importance,
        color='lightblue',
        alpha=0.8
    )
    ax.plot(
        window_timesteps,
        mean_importance,
        color='darkblue',
        linewidth=2,
        marker='o',
        markersize=4
    )
    
    ax.set_xlabel(f"Time Step (Center of Window) for angle {angle}", fontsize=12)
    ax.set_ylabel("Importance (Mean Absolute Change in Prediction)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add vertical lines for annotations if provided
    if annot_timesteps and machine_part == "All":
        annot_labels = ["Start-Declamping", "Start-Bending", "Start-Declamping", "End-Declamping"]
        y_max = np.max(mean_importance)
        y_range = y_max - np.min(mean_importance)
        
        for ts, label in zip(annot_timesteps, annot_labels):
            ax.axvline(ts, color="black", linestyle="--", alpha=0.7)
            # Place annotations at 85% of the y-axis height
            ax.annotate(
                label,
                xy=(ts, y_max * 0.85),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                rotation=90,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8, alpha=0.9)
            )
    
    if mandrel_extraction_annot_timesteps and machine_part == "All":
        ax.axvspan(
            mandrel_extraction_annot_timesteps[0],
            mandrel_extraction_annot_timesteps[1],
            color="blue",
            alpha=0.12,
            linewidth=0
        )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    window_importance_path =  window_importance_plots_dir/ f"size{occluded_window_size}-stride{stride}"
    window_importance_path.mkdir(parents=True, exist_ok=True)
    image_path = window_importance_path / f"window_importance_angle_{angle}.png"
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close("all")
    gc.collect()
    logger.info(f"Window importance plot saved for angle {angle}.")


def __create_sensor_plot_axis(ax: plt.Axes, sample_data: np.ndarray, 
                           sensor_names: list, colors: np.ndarray,
                           machine_part: str,
                           annot_timesteps: list = None,
                           mandrel_extraction_annot_timesteps: list = None) -> None:
    """Create sensor data plot on given axis."""
    cleaned_names = [name.replace("_mean", "") for name in sensor_names]
    main_timesteps = sample_data.shape[0]
    
    for i, color in enumerate(colors):
        ax.plot(sample_data[:, i], color=color, linewidth=2.5, alpha=0.85,
                label=cleaned_names[i], marker="o", markersize=3,
                markevery=max(1, main_timesteps // 20))
    
    ax.set_xlabel("Time Step", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sensor Value", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.set_facecolor("#f9f9f9")
    ax.set_xlim(0, main_timesteps-1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    if annot_timesteps and machine_part == "All":
        annot_labels = ["Start-Declamping", "Start-Bending", 
                       "Start-Declamping", "End-Declamping"]
        for ts, label in zip(annot_timesteps, annot_labels):
            ax.axvline(ts, color="black", linestyle="--", alpha=0.7)
            ax.annotate(label, xy=(ts, sample_data.max()), xytext=(0,10),
                       textcoords="offset points", ha="center", va="bottom",
                       fontsize=11, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8))
    
    if mandrel_extraction_annot_timesteps and machine_part == "All":
        ax.axvspan(mandrel_extraction_annot_timesteps[0],
                  mandrel_extraction_annot_timesteps[1],
                  color="blue", alpha=0.12, linewidth=0)


def __create_ig_heatmap_axis(ax: plt.Axes, attributions: np.ndarray,
                          sensor_names: list, target_name: str,
                          main_timesteps: int) -> None:
    """Create Integrated Gradients heatmap on given axis."""
    cleaned_names = [name.replace("_mean", "") for name in sensor_names]
    cleaned_names_reversed = list(reversed(cleaned_names))
    
    im = ax.imshow(attributions.T, cmap="magma", aspect="auto",
                  interpolation="nearest",
                  extent=[0, main_timesteps-1, 0, len(cleaned_names_reversed)])
    
    ax.set_yticks(np.arange(len(cleaned_names_reversed)) + 0.5)
    ax.set_yticklabels(cleaned_names_reversed)
    ax.set_xlim(0, main_timesteps-1)
    ax.set_xlabel("Time Step", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{target_name}", fontsize=12, fontweight="bold")
    ax.set_facecolor("white")
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(1.2)


def save_combined_ig_plot(ig_maps: list, sample_data: np.ndarray, 
                         sensor_names: list, target_feature_names: list,
                         saving_dir: Path, colors: np.ndarray, machine_part,
                         annot_timesteps: list = None,
                         mandrel_extraction_annot_timesteps: list = None,
                         figsize: tuple = (25, 3)) -> None:
    """Save combined Integrated Gradients plot with all features."""
    n_output_features = len(ig_maps)
    main_timesteps = sample_data.shape[0]
    
    n_rows = n_output_features + 1
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_rows), facecolor="white")
    gs = fig.add_gridspec(n_rows, 1, height_ratios=[2] + [1]*n_output_features, hspace=0.25)
    
    ax_main = fig.add_subplot(gs[0])
    __create_sensor_plot_axis(ax_main, sample_data, sensor_names, colors, machine_part,
                           annot_timesteps, mandrel_extraction_annot_timesteps)
    ax_main.legend(loc="upper left", bbox_to_anchor=(1.02, 1), 
                  frameon=True, fontsize=10)
    
    for idx, attributions in enumerate(ig_maps):
        ax = fig.add_subplot(gs[idx+1])
        target_name = target_feature_names[idx] if target_feature_names else f"Feature {idx}"
        __create_ig_heatmap_axis(ax, attributions, sensor_names, target_name, main_timesteps)
    
    plt.tight_layout()
    combined_path = saving_dir / "06_integrated_gradients" / "ig_combined.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_individual_ig_plots(ig_maps: list, sample_data: np.ndarray,
                            sensor_names: list, target_feature_names: list,
                            saving_dir: Path, colors: np.ndarray, machine_part,
                            annot_timesteps: list = None,
                            mandrel_extraction_annot_timesteps: list = None) -> None:
    """Save individual Integrated Gradients plots for each feature."""
    main_timesteps = sample_data.shape[0]
    
    for idx, attributions in enumerate(ig_maps):
        target_name = target_feature_names[idx] if target_feature_names else f"Feature_{idx}"
        feature_folder = saving_dir / "06_integrated_gradients" /target_name.replace(" ", "_")
        os.makedirs(feature_folder, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 10), facecolor="white")
        gs = fig.add_gridspec(2, 2, width_ratios=[0.88, 0.12],
                            height_ratios=[2, 1], hspace=0.3, wspace=0.05)
        
        ax_top = fig.add_subplot(gs[0, 0])
        __create_sensor_plot_axis(ax_top, sample_data, sensor_names, colors, machine_part,
                               annot_timesteps, mandrel_extraction_annot_timesteps)
        
        legend_ax = fig.add_subplot(gs[0, 1])
        legend_ax.axis('off')
        handles, labels = ax_top.get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc='upper left', fontsize=9,
                        frameon=True, borderpad=0.8, labelspacing=0.5,
                        handlelength=1.5, handletextpad=0.5, borderaxespad=0.5)
        
        ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
        __create_ig_heatmap_axis(ax_bottom, attributions, sensor_names, 
                              target_name, main_timesteps)
        
        cbar_ax = fig.add_subplot(gs[1, 1])
        cleaned_names_reversed = list(reversed([n.replace("_mean", "") for n in sensor_names]))
        im = ax_bottom.imshow(attributions.T, aspect="auto", cmap="magma",
                             interpolation="nearest",
                             extent=[0, main_timesteps-1, 0, len(cleaned_names_reversed)])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_linewidth(1.2)
        
        plt.tight_layout()
        indiv_path = feature_folder / "ig.png"
        fig.savefig(indiv_path, dpi=150, bbox_inches="tight")
        plt.close(fig)