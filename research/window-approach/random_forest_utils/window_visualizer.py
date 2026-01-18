import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_window_curve(
    df_sample: pd.DataFrame,
    importance_curve: np.ndarray,
    patch_size: int,
    stride: int,
    title: str,
) -> None:
    num_windows: int = len(importance_curve)
    window_starts = np.arange(num_windows) * stride
    window_ends = window_starts + patch_size

    fig, ax1 = plt.subplots(figsize=(18, 6))
    for col in df_sample.columns:
        ax1.plot(np.arange(len(df_sample)), df_sample[col], label=col)

    ax2 = ax1.twinx()
    importance_clean = np.nan_to_num(importance_curve, nan=0.0)
    max_imp = np.max(importance_clean) if np.max(importance_clean) > 0 else 1.0

    for start, end, imp in zip(window_starts, window_ends, importance_clean):
        alpha = 0.1 + 0.5 * (imp / max_imp)  
        ax2.fill_between([start, end], 0, imp, color="black", alpha=alpha)

    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Feature Value")
    ax2.set_ylabel("Importance (rescaled)")
    ax1.set_title(title)
    ax1.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_feature_heatmap(
    errors: np.ndarray,
    feature_names: List[str],
    patch_size: int,
    stride: int,
    title: str,
) -> None:
    num_features, num_patches = errors.shape
    window_centers = np.arange(num_patches) * stride + patch_size // 2

    plt.figure(figsize=(14, 6))
    im = plt.imshow(
        errors,
        aspect="auto",
        cmap="viridis",
        extent=(
            float(window_centers[0]), 
            float(window_centers[-1]), 
            0.0,
            float(num_features)
        ),
        origin="upper"
    )

    plt.colorbar(im, label="Permutation Error")
    plt.xlabel("Timestep (window center)")
    plt.ylabel("Features")
    plt.title(title)
    plt.yticks(ticks=np.arange(num_features), labels=feature_names)
    plt.tight_layout()
    plt.show()