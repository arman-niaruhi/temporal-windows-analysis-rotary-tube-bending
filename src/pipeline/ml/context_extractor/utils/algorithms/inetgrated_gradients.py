from captum.attr import IntegratedGradients
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.pipeline.ml.context_extractor.utils.helpers.plot_utils import save_combined_ig_plot, save_individual_ig_plots
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def __compute_integrated_gradients(model: nn.Module, X_sample: torch.Tensor,
                                n_output_features: int) -> list:
    """Compute Integrated Gradients attributions for all output features."""
    model.eval()
    ig_maps = []
    
    for idx in range(n_output_features):
        def forward_for_ig(x, target_idx=idx):
            pred, _ = model(x)
            return pred[:, :, target_idx].sum(dim=1)

        ig = IntegratedGradients(forward_for_ig)
        attributions, _ = ig.attribute(X_sample, return_convergence_delta=True)
        attributions = attributions.squeeze(0).cpu().detach().numpy()
        ig_maps.append(attributions)
    
    return ig_maps


def __save_ig_csvs(ig_maps: list, sensor_names: list, target_feature_names: list,
                saving_dir: Path) -> None:
    """Save Integrated Gradients attributions to CSV files."""
    cleaned_sensor_names = [name.replace("_mean", "") for name in sensor_names]
    
    for idx, attributions in enumerate(ig_maps):
        attr_df = pd.DataFrame(attributions, columns=cleaned_sensor_names)
        target_name = target_feature_names[idx] if target_feature_names else idx
        saving_path = saving_dir/ "06_integrated_gradients" 
        saving_path.mkdir(parents=True, exist_ok=True)
        csv_path = saving_path / f"ig_feature_{target_name}.csv"
        attr_df.to_csv(csv_path, index=False)


def save_integrated_gradients_combined(
    model: torch.nn.Module, X_sample: torch.Tensor,
    sensor_data: torch.Tensor, sensor_names: list[str],
    target_feature_names: list[str],
    saving_dir: Path,
    machine_part: str,
    annot_timesteps: list[int] = None,
    mandrel_extraction_annot_timesteps: list[int] = None,
    figsize_combined: tuple[int, int] = (25, 3),
):
    """Compute and save Integrated Gradients saliency maps."""
    model.eval()
    X_sample = X_sample.to(next(model.parameters()).device)
    
    with torch.no_grad():
        pred, _ = model(X_sample)
    
    n_output_features = pred.shape[2]
    sample_data = sensor_data[-1, :, :]
    colors = plt.cm.tab20(np.linspace(0, 1, len(sensor_names)))
    
    ig_maps = __compute_integrated_gradients(model, X_sample, n_output_features)
    
    __save_ig_csvs(ig_maps, sensor_names, target_feature_names, saving_dir)
    
    save_combined_ig_plot(ig_maps, sample_data, sensor_names, target_feature_names,
                         saving_dir, colors, machine_part, annot_timesteps,
                         mandrel_extraction_annot_timesteps, figsize_combined)
    
    save_individual_ig_plots(ig_maps, sample_data, sensor_names, target_feature_names,
                            saving_dir, colors, machine_part, annot_timesteps,
                            mandrel_extraction_annot_timesteps)