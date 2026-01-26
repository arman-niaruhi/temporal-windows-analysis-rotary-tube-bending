"""
Feature importance analysis derived from attention weights.

This module visualizes feature-level importance scores computed from
aggregated attention distributions.
The plots are used to analyze the relative contribution of input features
and their consistency across time.

All analyses in this module are attention-based and model-specific.
"""


import numpy as np
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def __plot_feature_importance_bar(importance_df, feature_names, output_path):
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



def __plot_attention_time_distribution(timestep_attention, output_path):
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



def __plot_attention_heatmap(timestep_attention, feature_names, output_path):
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



def __plot_cumulative_attention(timestep_attention, output_path):
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



def __plot_importance_rank_distribution(importance_df, output_path):
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
    __plot_feature_importance_bar(importance_df, feature_names, plot_paths["feature_importance"])
    
    plot_paths["time_distribution"] = output_dir / "02_attention_time_distribution.png"
    __plot_attention_time_distribution(timestep_attention, plot_paths["time_distribution"])
    
    plot_paths["heatmap"] = output_dir / "03_attention_heatmap.png"
    __plot_attention_heatmap(timestep_attention, feature_names, plot_paths["heatmap"])
    
    plot_paths["cumulative"] = output_dir / "04_cumulative_attention.png"
    __plot_cumulative_attention(timestep_attention, plot_paths["cumulative"])
    
    plot_paths["rank_distribution"] = output_dir / "05_importance_rank_distribution.png"
    __plot_importance_rank_distribution(importance_df, plot_paths["rank_distribution"])
    
    return plot_paths

