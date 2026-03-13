import numpy as np
import logging
import matplotlib.pyplot as plt
import gc
from matplotlib import rcParams

logger = logging.getLogger(__name__)


def _set_portable_font(size: int = 12) -> None:
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
    rcParams["font.size"] = size

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


def visualize_window_importance_heatmap(
    mean_importance_matrix: np.ndarray,
    window_importance_plots_dir,
    annot_timesteps,
    mandrel_extraction_annot_timesteps,
    process_part,
    occluded_window_size: int = 10,
    stride: int = 5,
):
    """
    Plot a global heatmap of window importance with:
    - y-axis: prediction angle index
    - x-axis: window center timestep
    """
    if mean_importance_matrix.size == 0:
        logger.warning("Window importance matrix is empty; skipping heatmap.")
        return

    n_angles, n_windows = mean_importance_matrix.shape
    window_centers = np.array(
        [i * stride + occluded_window_size // 2 for i in range(n_windows)]
    )

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))

    extent = [
        window_centers[0] if len(window_centers) else 0,
        window_centers[-1] if len(window_centers) else 0,
        0,
        n_angles - 1,
    ]
    im = ax.imshow(
        mean_importance_matrix,
        cmap="YlOrRd",
        aspect="auto",
        origin="upper",
        extent=extent,
    )

    ax.set_xlabel("Time Step (Center of Window)", fontsize=12)
    ax.set_ylabel("Angle Index", fontsize=12)
    ax.set_yticklabels(
        [f"{i + 1}" for i in reversed(range(n_angles - 1))], fontsize=5
    )
    ax.set_title("Window Importance Heatmap", fontsize=14, fontweight="bold")

    if annot_timesteps and process_part == "All":
        for ts in annot_timesteps:
            ax.axvline(ts, color="black", linestyle="--", alpha=0.6, linewidth=1)

    if mandrel_extraction_annot_timesteps and process_part == "All":
        ax.axvspan(
            mandrel_extraction_annot_timesteps[0],
            mandrel_extraction_annot_timesteps[1],
            color="blue",
            alpha=0.12,
            linewidth=0,
        )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Importance", rotation=270, labelpad=15)

    plt.tight_layout()

    window_importance_path = (
        window_importance_plots_dir / f"size{occluded_window_size}-stride{stride}"
    )
    window_importance_path.mkdir(parents=True, exist_ok=True)
    image_path = window_importance_path / "window_importance_heatmap.png"
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    gc.collect()
    logger.info("Window importance heatmap saved.")


def visualize_window_importance_with_sensors(
    sensor_data: np.ndarray,
    sensor_names: list[str],
    mean_importance_matrix: np.ndarray,
    window_importance_plots_dir,
    annot_timesteps,
    mandrel_extraction_annot_timesteps,
    process_part,
    occluded_window_size: int = 10,
    stride: int = 5,
):
    """
    Attention-style visualization:
    - Top: sensor lines over time
    - Bottom: window-importance heatmap (angles x timesteps)
    """
    if mean_importance_matrix.size == 0 or sensor_data.size == 0:
        logger.warning("Empty data for combined window-importance plot; skipping.")
        return

    _set_portable_font(size=12)

    sample_data = sensor_data[-1, :, :]
    main_timesteps = sample_data.shape[0]
    n_angles, n_windows = mean_importance_matrix.shape
    if n_windows == 0:
        logger.warning("No windows available for combined window-importance plot; skipping.")
        return

    # Resample window-based importance (window centers) to per-timestep grid
    window_centers = np.array(
        [i * stride + occluded_window_size // 2 for i in range(n_windows)],
        dtype=float,
    )
    time_axis = np.arange(main_timesteps, dtype=float)
    heatmap_data = np.zeros((n_angles, main_timesteps), dtype=float)

    for angle_idx in range(n_angles):
        heatmap_data[angle_idx] = np.interp(
            time_axis,
            window_centers,
            mean_importance_matrix[angle_idx],
            left=mean_importance_matrix[angle_idx, 0],
            right=mean_importance_matrix[angle_idx, -1],
        )

    fig = plt.figure(figsize=(25, 12), facecolor="white")
    fig.clf()
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.25, wspace=0.3)
    ax_main = fig.add_subplot(gs[0])
    ax_heatmap = fig.add_subplot(gs[1])

    cleaned_feature_names = [name.replace("_mean", "") for name in sensor_names]
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

    if annot_timesteps and (process_part == "All"):
        annot_labels = [
            "Start-Clamping",
            "Start-Bending",
            "Start-Declamping",
            "End-Declamping",
        ]
        for ts, label in zip(annot_timesteps, annot_labels):
            ax_main.axvline(ts, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
            ax_main.annotate(
                label,
                xy=(ts, sample_data[:, :].max()),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8),
            )

    if mandrel_extraction_annot_timesteps and (process_part == "All"):
        ax_main.axvspan(
            mandrel_extraction_annot_timesteps[0],
            mandrel_extraction_annot_timesteps[1],
            color="blue",
            alpha=0.12,
            linewidth=0,
            zorder=0.5,
        )

    ax_main.set_xlim(0, main_timesteps - 1)
    ax_main.set_facecolor("#f9f9f9")

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

    im = ax_heatmap.imshow(
        heatmap_data,
        aspect="auto",
        origin="upper",
        cmap="magma",
        interpolation="bilinear",
        extent=[0, main_timesteps - 1, 0, n_angles - 1],
    )

    
    ax_heatmap.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
    ax_heatmap.set_ylabel("Angle Index", fontsize=9, fontweight="bold", labelpad=10)
    ax_heatmap.set_xlim(0, main_timesteps - 1)
    ax_heatmap.set_facecolor("white")
    ax_heatmap.spines["top"].set_visible(False)
    ax_heatmap.spines["right"].set_visible(False)
    ax_heatmap.spines["left"].set_linewidth(1.2)
    ax_heatmap.spines["bottom"].set_linewidth(1.2)
    ax_heatmap.spines["left"].set_color("#333333")
    ax_heatmap.spines["bottom"].set_color("#333333")
    ax_heatmap.set_yticks(np.linspace(0, n_angles - 1, 5))
    ax_heatmap.set_yticklabels(
        [str(int(v)) for v in np.linspace(n_angles - 1, 0,  5)],
        fontsize=9
    )

    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.9, pad=0.02)
    cbar.set_label("Importance", fontsize=11, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(1.2)

    plt.tight_layout()
    pos_main = ax_main.get_position()
    pos_heat = ax_heatmap.get_position()
    ax_heatmap.set_position([pos_heat.x0, pos_heat.y0, pos_main.width, pos_heat.height])
    cbar.ax.set_position([pos_main.x0 + pos_main.width + 0.02, pos_heat.y0, 0.015, pos_heat.height])

    window_importance_path = (
        window_importance_plots_dir / f"size{occluded_window_size}-stride{stride}"
    )
    window_importance_path.mkdir(parents=True, exist_ok=True)
    image_path = window_importance_path / "window_importance_with_sensors.png"
    fig.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    logger.info("Combined window importance (sensor + heatmap) plot saved.")
