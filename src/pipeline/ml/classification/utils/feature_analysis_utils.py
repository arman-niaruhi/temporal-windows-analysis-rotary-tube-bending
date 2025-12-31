import logging
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.pipeline.ml.classification.utils.model import LSTMSequenceClassifier
from src.pipeline.preprocessing.loader import DataLoader

logger = logging.getLogger(__name__)


def permutation_importance_sequence(model: LSTMSequenceClassifier, data_loader: DataLoader, device: torch.device, n_repeats: int = 10):
    """
    Calculate permutation importance for sequence model with padding.
    Works directly with the DataLoader.
    
    Args:
        model (LSTMSequenceClassifier): Trained sequence classification model.
        data_loader (DataLoader): DataLoader providing the input data.
        device (torch.device): Device to run computations on.
        n_repeats (int): Number of permutation repeats for averaging.
    
    Returns:
        importances (np.ndarray): Array of feature importance scores.
        importance_std (np.ndarray): Standard deviation of importance scores.
    """
    model.eval()

    logger.info("Calculating baseline accuracy (permutation importance)")
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch, mask_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            outputs = model(X_batch)
            predictions = outputs.argmax(dim=-1)

            valid_mask = (y_batch != -1) & (mask_batch == 1)
            correct += ((predictions == y_batch) & valid_mask).sum().item()
            total += valid_mask.sum().item()

    baseline_accuracy = correct / total
    logger.info("Baseline accuracy: %.4f", baseline_accuracy)

    for X_batch, _, _ in data_loader:
        num_features = X_batch.shape[2]
        break

    importances = []
    importance_std = []

    for feature_idx in range(num_features):
        logger.info("Processing feature %d / %d", feature_idx + 1, num_features)
        feature_scores = []

        for repeat in range(n_repeats):
            logger.debug("  Repeat %d / %d", repeat + 1, n_repeats)
            correct = 0
            total = 0

            with torch.no_grad():
                for X_batch, y_batch, mask_batch in data_loader:
                    X_permuted = X_batch.clone()

                    perm_values = X_batch[:, :, feature_idx].flatten()
                    perm_indices = torch.randperm(perm_values.shape[0])
                    X_permuted[:, :, feature_idx] = perm_values[perm_indices].reshape(
                        X_batch.shape[0], X_batch.shape[1]
                    )

                    X_permuted = X_permuted.to(device)
                    y_batch = y_batch.to(device)
                    mask_batch = mask_batch.to(device)

                    outputs = model(X_permuted)
                    predictions = outputs.argmax(dim=-1)

                    valid_mask = (y_batch != -1) & (mask_batch == 1)
                    correct += ((predictions == y_batch) & valid_mask).sum().item()
                    total += valid_mask.sum().item()

            permuted_accuracy = correct / total
            feature_scores.append(baseline_accuracy - permuted_accuracy)

        mean_importance = np.mean(feature_scores)
        std_importance = np.std(feature_scores)

        importances.append(mean_importance)
        importance_std.append(std_importance)

        logger.info(
            "Feature %d importance: %.4f ± %.4f",
            feature_idx,
            mean_importance,
            std_importance,
        )

    return np.array(importances), np.array(importance_std)


def gradient_importance_sequence(model: LSTMSequenceClassifier, data_loader: DataLoader, device: torch.device, n_batches: int = 10):
    """
    Calculate Gradient × Input importance for a sequence model.
    Averages contributions over valid timesteps only.
    
    Args:
        model (LSTMSequenceClassifier): Trained sequence classification model.
        data_loader (DataLoader): DataLoader providing input sequences.
        device (torch.device): Device to run computations on.
        n_batches (int): Number of batches to process for averaging.
    
    Returns:
        importances (np.ndarray): Array of feature importance scores.
    """
    import numpy as np

    model.eval()
    logger.info("Calculating Gradient × Input importance")

    all_gradients = []
    batch_count = 0

    for X_batch, y_batch, mask_batch in data_loader:
        if batch_count >= n_batches:
            break

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)
        X_batch.requires_grad = True

        outputs = model(X_batch)
        model.zero_grad()

        # Valid timesteps mask
        valid_mask = (y_batch != -1) & (mask_batch == 1)
        loss = outputs[valid_mask].max(dim=-1)[0].sum()
        loss.backward()

        # Gradient × Input
        gradients = X_batch.grad  # shape: [batch, seq_len, features]
        grad_x_input = (gradients * X_batch).detach()  # element-wise product

        for i in range(X_batch.shape[0]):
            valid_indices = torch.where(valid_mask[i])[0]
            if len(valid_indices) > 0:
                # Average over valid timesteps
                grad_mean = grad_x_input[i, valid_indices, :].mean(dim=0).cpu().numpy()
                all_gradients.append(grad_mean)

        batch_count += 1

    # Average over all samples
    importances = np.mean(all_gradients, axis=0)
    logger.info("Gradient × Input importance computed")

    return importances


def integrated_gradients_importance(model: LSTMSequenceClassifier, data_loader: DataLoader, device: torch.device, n_samples: int = 50, steps: int = 30):
    """
    Calculate integrated gradients importance for sequence model.
    
    Args:
        model (LSTMSequenceClassifier): Trained sequence classification model.
        data_loader (DataLoader): DataLoader providing the input data.
        device (torch.device): Device to run computations on.
        n_samples (int): Number of samples to process for averaging.
        steps (int): Number of interpolation steps for integrated gradients.
    
    Returns:
        importances (np.ndarray): Array of feature importance scores.
    """
    model.eval()
    logger.info("Calculating integrated gradients importance")

    all_importances = []
    sample_count = 0

    for X_batch, y_batch, mask_batch in data_loader:
        if sample_count >= n_samples:
            break

        X_batch = X_batch.to(device)
        mask_batch = mask_batch.to(device)

        for i in range(X_batch.shape[0]):
            if sample_count >= n_samples:
                break

            valid_indices = torch.where(mask_batch[i] == 1)[0]
            if len(valid_indices) == 0:
                continue

            X_sample = X_batch[i : i + 1].clone()
            baseline = torch.zeros_like(X_sample)
            accumulated_grads = torch.zeros_like(X_sample)

            for alpha in np.linspace(0, 1, steps):
                X_interp = (
                    (baseline + alpha * (X_sample - baseline))
                    .clone()
                    .detach()
                    .requires_grad_(True)
                )

                outputs = model(X_interp)
                pred_class = outputs.argmax(dim=-1)

                model.zero_grad()
                outputs[0, pred_class].sum().backward()
                accumulated_grads += X_interp.grad

            avg_grads = accumulated_grads / steps
            integrated_grads = (X_sample - baseline) * avg_grads

            feature_importance = (
                integrated_grads[0, valid_indices, :]
                .abs()
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
            )

            all_importances.append(feature_importance)
            sample_count += 1

    importances = np.mean(all_importances, axis=0)
    logger.info("Integrated gradients importance computed")

    return importances


def occlusion_importance(
    model: LSTMSequenceClassifier, data_loader: DataLoader, device: torch.device, occlusion_value: float = 0.0
):
    """
    Occlusion-based importance.
    
    Args:
        model (LSTMSequenceClassifier): Trained sequence classification model.
        data_loader (DataLoader): DataLoader providing the input data.
        device (torch.device): Device to run computations on.
        occlusion_value (float): Value to use for occlusion.
        
    Returns:
        importances (np.ndarray): Array of feature importance scores.   
    """
    model.eval()
    logger.info("Calculating baseline accuracy (occlusion)")

    baseline_correct = 0
    baseline_total = 0

    with torch.no_grad():
        for X_batch, y_batch, mask_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            outputs = model(X_batch)
            predictions = outputs.argmax(dim=-1)

            valid_mask = (y_batch != -1) & (mask_batch == 1)
            baseline_correct += ((predictions == y_batch) & valid_mask).sum().item()
            baseline_total += valid_mask.sum().item()

    baseline_accuracy = baseline_correct / baseline_total
    logger.info("Baseline accuracy: %.4f", baseline_accuracy)

    for X_batch, _, _ in data_loader:
        num_features = X_batch.shape[2]
        break

    importances = []

    for feature_idx in range(num_features):
        logger.info("Occluding feature %d / %d", feature_idx + 1, num_features)
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch, mask_batch in data_loader:
                X_occluded = X_batch.clone()
                X_occluded[:, :, feature_idx] = occlusion_value

                X_occluded = X_occluded.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)

                outputs = model(X_occluded)
                predictions = outputs.argmax(dim=-1)

                valid_mask = (y_batch != -1) & (mask_batch == 1)
                correct += ((predictions == y_batch) & valid_mask).sum().item()
                total += valid_mask.sum().item()

        occluded_accuracy = correct / total
        importance = baseline_accuracy - occluded_accuracy
        importances.append(importance)

        logger.info("Feature %d importance: %.4f", feature_idx, importance)

    return np.array(importances)


def dropout_importance(model: LSTMSequenceClassifier, data_loader: DataLoader, device: torch.device, n_repeats: int = 20, dropout_rate: float = 0.5):
    """
    Dropout-based importance.
    
    Args:
        model (LSTMSequenceClassifier): Trained sequence classification model.
        data_loader (DataLoader): DataLoader providing the input data.
        device (torch.device): Device to run computations on.
        n_repeats (int): Number of dropout repeats for averaging.
        
    Returns:
        importances (np.ndarray): Array of feature importance scores.
    """
    model.eval()
    logger.info("Calculating dropout-based importance")

    for X_batch, _, _ in data_loader:
        num_features = X_batch.shape[2]
        break

    importances = []

    for feature_idx in range(num_features):
        logger.info("Testing feature %d / %d", feature_idx + 1, num_features)
        prediction_variances = []

        with torch.no_grad():
            for X_batch, _, mask_batch in data_loader:
                X_batch = X_batch.to(device)
                mask_batch = mask_batch.to(device)

                batch_predictions = []

                for _ in range(n_repeats):
                    X_dropped = X_batch.clone()
                    dropout_mask = (
                        torch.rand(X_batch.shape[0], X_batch.shape[1], device=device)
                        > dropout_rate
                    )
                    X_dropped[:, :, feature_idx] *= dropout_mask.float()

                    outputs = model(X_dropped)
                    batch_predictions.append(outputs.softmax(dim=-1).cpu().numpy())

                variance = np.var(np.array(batch_predictions), axis=0).mean()
                prediction_variances.append(variance)

        importance = np.mean(prediction_variances)
        importances.append(importance)

        logger.info("Feature %d importance: %.6f", feature_idx, importance)

    return np.array(importances)


def feature_ablation_importance(model: LSTMSequenceClassifier, data_loader: DataLoader, device: torch.device):
    """
    Feature ablation importance.
    
    Args:
        model (LSTMSequenceClassifier): Trained sequence classification model.
        data_loader (DataLoader): DataLoader providing the input data.
        device (torch.device): Device to run computations on.

    Returns:
        importances (np.ndarray): Array of feature importance scores.
    """
    model.eval()
    logger.info("Calculating baseline accuracy (ablation)")

    baseline_correct = 0
    baseline_total = 0

    with torch.no_grad():
        for X_batch, y_batch, mask_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            outputs = model(X_batch)
            predictions = outputs.argmax(dim=-1)

            valid_mask = (y_batch != -1) & (mask_batch == 1)
            baseline_correct += ((predictions == y_batch) & valid_mask).sum().item()
            baseline_total += valid_mask.sum().item()

    baseline_accuracy = baseline_correct / baseline_total
    logger.info("Baseline accuracy: %.4f", baseline_accuracy)

    for X_batch, _, _ in data_loader:
        num_features = X_batch.shape[2]
        break

    importances = []

    for feature_idx in range(num_features):
        logger.info("Ablating feature %d / %d", feature_idx + 1, num_features)
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch, mask_batch in data_loader:
                X_ablated = X_batch.clone()
                X_ablated[:, :, feature_idx] = 0.0

                X_ablated = X_ablated.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)

                outputs = model(X_ablated)
                predictions = outputs.argmax(dim=-1)

                valid_mask = (y_batch != -1) & (mask_batch == 1)
                correct += ((predictions == y_batch) & valid_mask).sum().item()
                total += valid_mask.sum().item()

        ablated_accuracy = correct / total
        importance = baseline_accuracy - ablated_accuracy
        importances.append(importance)

        logger.info("Feature %d importance: %.4f", feature_idx, importance)

    return np.array(importances)


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
