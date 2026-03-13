"""
Attention-based visualizations for temporal and feature-level analysis.

This module contains all plots that directly visualize attention weights,
including heatmaps and line-based attention overlays.
These figures are used to interpret how the model allocates attention
across time steps and sensor features.

The module is intentionally limited to attention mechanisms and does not
include gradient-based attribution methods.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams


def _set_portable_font(size: int = 12) -> None:
    """Prefer Arial when available, with safe cross-platform fallbacks."""
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
    rcParams["font.size"] = size


def generate_final_attention_plot(
    model: nn.Module,
    plot_X: torch.Tensor,
    springback: torch.Tensor,
    experiment_config: torch.Tensor,
    X_val: torch.Tensor,
    sensor_names: list,
    machine_part: str,
    attention_lines_dir: Path,
    annot_timesteps: list,
    mandrel_extraction_annot_timesteps: list,
    target_feature_names: list | None = None,
) -> None:
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
        _, final_attn = model(plot_X, springback, experiment_config)
        if final_attn is None:
            return
        final_attn_mean = final_attn.mean(0).cpu().numpy()

    
    sensor_data=X_val
    attn_mean=final_attn_mean
    sample_idx=-1
    figsize: tuple=(20, 10)
    _set_portable_font(size=12)

    cleaned_feature_names = [name.replace("_mean", "") for name in sensor_names]

    if attn_mean.ndim == 3:
        for feat_idx in range(attn_mean.shape[0]):
            feat_dir = attention_lines_dir / f"feature_{feat_idx:02d}"
            feat_dir.mkdir(parents=True, exist_ok=True)
            plot_attention_lines_with_sensors(
                sensor_data,
                sensor_names,
                attn_mean[feat_idx],
                machine_part,
                feat_dir,
                annot_timesteps,
                mandrel_extraction_annot_timesteps,
                sample_idx=sample_idx,
                figsize=figsize,
            )
        return

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
    _set_portable_font(size=12)

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
            "Start-Clamping",
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
    
