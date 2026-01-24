"""
Utility functions shared across all plotting modules.

This module contains small, stateless helper functions used to prepare data
for visualization (e.g., extracting batches, computing axis limits).
It intentionally does not contain any figure-specific plotting logic.

This separation ensures that figure-generation code remains concise,
readable, and directly traceable to the figures reported in the paper.
"""

import torch
from torch.utils.data import DataLoader

def get_plot_batch(loader: DataLoader, device: torch.device):
    x, y, springback, experiment_config = next(iter(loader))
    return (
        x.to(device),
        y.to(device),
        springback.to(device),
        experiment_config.to(device),
    )

def compute_plot_limits(Y_val: torch.Tensor) -> list[tuple[float, float]]:
    """Compute per-feature y-limits for consistent plotting."""
    y_all = Y_val.detach().cpu().numpy()
    limits: list[tuple[float, float]] = []
    for feat in range(y_all.shape[-1]):
        feat_vals = y_all[:, :, feat]
        feat_min = float(feat_vals.min())
        feat_max = float(feat_vals.max())
        span = feat_max - feat_min
        margin = span * 0.1 if span > 0 else 1.0
        limits.append((feat_min - margin, feat_max + margin))
    return limits
