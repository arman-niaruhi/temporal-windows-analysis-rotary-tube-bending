import numpy as np
import logging
import matplotlib.pyplot as plt
import gc

logger = logging.getLogger(__name__)

def visualize_window_importance(
    angle: int,
    feature_names: list[str],
    mean_importance,
    annot_timesteps,
    window_importance_plots_dir,
    mandrel_extraction_annot_timesteps,
    process_part,
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
    if annot_timesteps and process_part == "All":
        annot_labels = ["Start-Declamping", "Start-Bending", "Start-Declamping", "End-Declamping"]
        y_max = np.max(mean_importance)
        
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
    
    if mandrel_extraction_annot_timesteps and process_part == "All":
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

