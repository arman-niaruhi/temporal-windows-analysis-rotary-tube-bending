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
    x, y, springback = next(iter(loader))
    return x.to(device), y, springback

def compute_plot_limits(Y_val: torch.Tensor) -> tuple:
    """Compute global y-limits for consistent plotting."""
    y_all = Y_val[:, :, 0].cpu().numpy()
    global_ymin, global_ymax = y_all.min(), y_all.max()
    margin = (global_ymax - global_ymin) * 0.1
    return (global_ymin - margin, global_ymax + margin)

