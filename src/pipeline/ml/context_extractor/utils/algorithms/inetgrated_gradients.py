from captum.attr import IntegratedGradients
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.pipeline.ml.context_extractor.utils.plots.plot_integrated_gradients import save_individual_ig_plots
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def __compute_integrated_gradients(
    model: nn.Module, 
    X_sample: torch.Tensor, 
    springback_sample: torch.Tensor,
    experiment_config: torch.Tensor,
    n_output_features: int
) -> list:
    """
    Compute Integrated Gradients attributions for all output features.
    
    Note: springback_sample is held constant during integration. Only X_sample
    is integrated from baseline to actual input.
    """
    model.eval()
    ig_maps = []
    
    for idx in range(n_output_features):
        # Create wrapper that only takes x as input (springback is fixed)
        def forward_for_ig(x):
            # CRITICAL FIX: Expand springback to match batch size of interpolated samples
            # x will have shape (n_steps, seq_len, features) where n_steps is typically 50
            # springback_sample has shape (1, 1)
            batch_size = x.shape[0]
            springback_base = springback_sample
            if springback_base.dim() == 1:
                springback_base = springback_base.unsqueeze(0)
            springback_expanded = springback_base.expand(batch_size, -1)

            config_expanded = None
            if experiment_config is not None:
                config_base = experiment_config
                if config_base.dim() == 1:
                    config_base = config_base.unsqueeze(0)
                config_expanded = config_base.expand(batch_size, -1)
            
            pred, _ = model(x, springback_expanded, config_expanded)
            # Sum over prediction timesteps to get single output per sample
            return pred[:, :, idx].sum(dim=1)
        
        ig = IntegratedGradients(forward_for_ig)
        # n_steps controls how many interpolations between baseline and input
        # More steps = more accurate but slower. Default is 50, which is good.
        attributions, _ = ig.attribute(
            X_sample, 
            n_steps=50,  # You can change this (e.g., 100 for more accuracy)
            return_convergence_delta=True
        )
        attributions = attributions.squeeze(0).cpu().detach().numpy()
        ig_maps.append(attributions)
    
    return ig_maps


def save_integrated_gradients_combined(
    model: torch.nn.Module, 
    X_sample: torch.Tensor,
    springback_sample: torch.Tensor,
    experiment_config: torch.Tensor,
    sensor_data: torch.Tensor,
    sensor_names: list[str],
    target_feature_names: list[str],
    saving_dir: Path,
    process_part: str,
    annot_timesteps: list[int] = None,
    mandrel_extraction_annot_timesteps: list[int] = None,
    figsize_combined: tuple[int, int] = (25, 3),
):
    """Compute and save Integrated Gradients saliency maps."""
    model.eval()
    device = next(model.parameters()).device

    if X_sample.dim() == 2:
        X_sample = X_sample.unsqueeze(0)
    if springback_sample is not None and springback_sample.dim() == 0:
        springback_sample = springback_sample.unsqueeze(0)
    if springback_sample is not None and springback_sample.dim() == 1:
        springback_sample = springback_sample.unsqueeze(-1)
    if experiment_config is not None and experiment_config.dim() == 1:
        experiment_config = experiment_config.unsqueeze(0)

    X_sample = X_sample.to(device)
    springback_sample = springback_sample.to(device)
    if experiment_config is not None:
        experiment_config = experiment_config.to(device)

    with torch.no_grad():
        pred, _ = model(X_sample, springback_sample, experiment_config)
    
    n_output_features = pred.shape[2]
    sample_data = sensor_data[45, :, :]
    colors = plt.cm.tab20(np.linspace(0, 1, len(sensor_names)))
    
    # FIXED: Corrected argument order - springback_sample before n_output_features
    ig_maps = __compute_integrated_gradients(
        model,
        X_sample,
        springback_sample,
        experiment_config,
        n_output_features,
    )
    
    save_individual_ig_plots(
        ig_maps, 
        sample_data, 
        sensor_names, 
        target_feature_names,
        saving_dir, 
        colors, 
        process_part, 
        annot_timesteps,
        mandrel_extraction_annot_timesteps
    )
