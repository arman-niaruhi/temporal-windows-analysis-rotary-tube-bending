"""
Gradient-based attribution analysis using Integrated Gradients.

This module contains visualization routines for Integrated Gradients (IG),
including per-feature attribution plots and combined attribution heatmaps.
These figures provide an alternative interpretability perspective that
complements attention-based analyses.

The module is fully independent of attention-weight visualizations.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import os
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def __create_sensor_plot_axis(ax: Axes, sample_data: np.ndarray, 
                           sensor_names: list, colors: np.ndarray,
                           machine_part: str,
                           annot_timesteps: list,
                           mandrel_extraction_annot_timesteps) -> None:
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



def __create_ig_heatmap_axis(ax: Axes, attributions: np.ndarray,
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
                         annot_timesteps: list,
                         mandrel_extraction_annot_timesteps: list,
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
        target_name = target_feature_names[idx-1] if target_feature_names else f"Feature {idx}"
        __create_ig_heatmap_axis(ax, attributions, sensor_names, target_name, main_timesteps)
    
    plt.tight_layout()
    combined_path = saving_dir / "ig_combined.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def save_individual_ig_plots(ig_maps: list, sample_data: np.ndarray,
                            sensor_names: list, target_feature_names: list,
                            saving_dir: Path, colors: np.ndarray, machine_part,
                            annot_timesteps: list,
                            mandrel_extraction_annot_timesteps: list) -> None:
    """Save individual Integrated Gradients plots for each feature."""
    main_timesteps = sample_data.shape[0]
    
    for idx, attributions in enumerate(ig_maps):
        target_name = target_feature_names[idx-1] if target_feature_names else f"Feature_{idx}"
        feature_folder = saving_dir /target_name.replace(" ", "_")
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