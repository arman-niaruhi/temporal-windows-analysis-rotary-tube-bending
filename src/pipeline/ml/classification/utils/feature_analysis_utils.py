import logging
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

from captum.attr import IntegratedGradients


def compute_feature_mean_baseline(data_loader, device):
    all_feats = []

    for X, _, mask in data_loader:
        X = X.to(device)
        mask = mask.to(device)  # (B, T, 1)

        # remove the singleton dimension
        mask = mask.squeeze(-1)  # (B, T)

        # flatten batch and time
        X_flat = X.reshape(-1, X.size(-1))        # (B*T, F)
        mask_flat = mask.reshape(-1) == 1          # (B*T,)

        all_feats.append(X_flat[mask_flat])

    return torch.cat(all_feats, dim=0).mean(dim=0)


class LSTMWrapper(torch.nn.Module):
    def __init__(self, model, target_class):
        super().__init__()
        self.model = model
        self.target_class = target_class

    def forward(self, x):
        # x: (B, T, F)
        logits = self.model(x)  # (B, T, C)
        return logits[:, :, self.target_class].sum(dim=1)


def captum_classwise_ig(
    model,
    data_loader,
    device,
    target_class: int,
    n_samples: int = 100,
):
    wrapper = LSTMWrapper(model, target_class).to(device)
    ig = IntegratedGradients(wrapper)

    all_attr = []
    sample_count = 0

    baseline_feat = compute_feature_mean_baseline(data_loader, device)

    for X, y, mask in data_loader:
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)

        for i in range(X.size(0)):
            if y[i].eq(target_class).any() is False:
                continue

            baseline = baseline_feat.view(1, 1, -1).expand_as(X[i:i+1])
            attr = ig.attribute(
                X[i:i+1],
                baselines=baseline,
                n_steps=50,
            )

            # mask padded timesteps
            valid_idx = mask[i] == 1
            attr = attr[0, valid_idx, :].abs().mean(dim=0)

            all_attr.append(attr.cpu().numpy())
            sample_count += 1

            if sample_count >= n_samples:
                break

        if sample_count >= n_samples:
            break

    return np.mean(all_attr, axis=0)


def plot_feature_importance(
    analyze_features_result_path: str, importances: np.ndarray, std: np.ndarray = None, feature_names: list = None, method_name: str = "Permutation"
):
    """Plot feature importance with error bars.
    
    Args:
        analyze_features_result_path (str): Directory to save the plot.
        importances (np.ndarray): Array of feature importance scores.
        std (np.ndarray, optional): Standard deviation of importance scores. Defaults to None.
        feature_names (list, optional): List of feature names. Defaults to None.
        method_name (str, optional): Name of the importance method. Defaults to "Permutation".
    
    Returns:
        None: Saves the plot to the specified directory.
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    plt.figure(figsize=(10, max(6, len(importances) * 0.3)))

    if std is not None:
        sorted_std = std[sorted_idx]
        plt.barh(range(len(sorted_importances)), sorted_importances, xerr=sorted_std)
    else:
        plt.barh(range(len(sorted_importances)), sorted_importances)

    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel("Importance Score")
    plt.title(f"Feature Importance ({method_name} Method)")
    plt.tight_layout()
    
    store_path = os.path.join(analyze_features_result_path, f"feature_importance_{method_name.lower()}.png")
    plt.savefig(
        store_path, dpi=300, bbox_inches="tight"
    )


def compare_methods(analyze_features_result_path, method_dict, feature_names=None):
    """
    Compare multiple importance methods side by side.
    method_dict: {'Method Name': importance_array, ...}
    
    Args:
        method_dict (dict): Dictionary mapping method names to importance arrays.
        feature_names (list, optional): List of feature names. Defaults to None.
    Returns:
        None: Saves comparison plots to the current directory.
    """
    if feature_names is None:
        first_imp = list(method_dict.values())[0]
        feature_names = [f"Feature {i}" for i in range(len(first_imp))]

    normalized_methods = {}
    for method_name, importances in method_dict.items():
        imp_min, imp_max = importances.min(), importances.max()
        normalized = (importances - imp_min) / (imp_max - imp_min + 1e-10)
        normalized_methods[method_name] = normalized

    x = np.arange(len(feature_names))
    width = 0.8 / len(method_dict)

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (method_name, norm_imp) in enumerate(normalized_methods.items()):
        offset = width * (i - len(method_dict) / 2 + 0.5)
        ax.bar(x + offset, norm_imp, width, label=method_name, alpha=0.8)

    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Normalized Importance", fontsize=12)
    ax.set_title(
        "Feature Importance - All Methods Comparison", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    output_file = Path(analyze_features_result_path) / "feature_importance_all_methods_comparison.png"
    plt.tight_layout()
    plt.savefig(
        output_file, dpi=300, bbox_inches="tight"
    )

    importance_matrix = np.array([imp for imp in normalized_methods.values()])

    fig, ax = plt.subplots(figsize=(12, len(method_dict) * 1.2))
    sns.heatmap(
        importance_matrix,
        xticklabels=feature_names,
        yticklabels=list(method_dict.keys()),
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Normalized Importance"},
    )
    plt.title(
        "Feature Importance Heatmap - All Methods", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Methods", fontsize=12)
    plt.tight_layout()
    
    output_file = Path(analyze_features_result_path) / "feature_importance_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    if len(method_dict) > 1:
        method_names = list(method_dict.keys())
        n_methods = len(method_names)
        correlation_matrix = np.zeros((n_methods, n_methods))

        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                correlation_matrix[i, j] = np.corrcoef(
                    method_dict[method1], method_dict[method2]
                )[0, 1]

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            correlation_matrix,
            xticklabels=method_names,
            yticklabels=method_names,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            vmin=-1,
            vmax=1,
            center=0,
            cbar_kws={"label": "Correlation"},
        )
        plt.title("Method Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        output_file = Path(analyze_features_result_path) / "method_correlation.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
