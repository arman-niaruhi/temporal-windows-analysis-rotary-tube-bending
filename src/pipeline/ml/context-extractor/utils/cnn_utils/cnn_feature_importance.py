import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def permutation_feature_importance(model, test_dataset, num_samples=50):
    """
    Compute permutation feature importance by measuring performance drop
    when shuffling each feature
    """
    model.eval()
    criterion = nn.MSELoss()

    # Baseline performance
    baseline_loss = 0
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            x, y = test_dataset[i]
            x, y = x.unsqueeze(0), y.unsqueeze(0)
            output = model(x)
            loss = criterion(output, y)
            baseline_loss += loss.item()
    baseline_loss /= num_samples

    # Feature importance scores
    feature_importance = []
    num_features = test_dataset[0][0].shape[1]

    for feature_idx in range(num_features):
        permuted_loss = 0
        with torch.no_grad():
            for i in range(min(num_samples, len(test_dataset))):
                x, y = test_dataset[i]
                x_permuted = x.clone()
                # Shuffle the specific feature across time steps
                x_permuted[:, feature_idx] = x[torch.randperm(x.shape[0]), feature_idx]

                x_permuted, y = x_permuted.unsqueeze(0), y.unsqueeze(0)
                output = model(x_permuted)
                loss = criterion(output, y)
                permuted_loss += loss.item()

        permuted_loss /= num_samples
        importance = permuted_loss - baseline_loss  # Higher = more important
        feature_importance.append(importance)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_features), feature_importance)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance (Loss Increase)")
    plt.title("Permutation Feature Importance")
    plt.show()

    return feature_importance


# Complete training and evaluation pipeline
def full_pipeline(X, Y, model, model_path, num_epochs, lr):
    # Your existing data setup
    X_tensor = torch.tensor(X, dtype=torch.float32)  # (100, 88, 17)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)  # (100, 5, 4)

    dataset = TensorDataset(X_tensor, Y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print("Training CNN Model...")
    model, train_losses, test_losses = model.train_cnn_model(
        train_loader, test_loader, num_epochs=num_epochs, lr=lr, save_path=model_path
    )

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.title("Training History")
    plt.legend()
    plt.show()

    # Visualize feature importance
    print("Visualizing Feature Importance...")
    importance_maps = model.visualize_cnn_importance(test_dataset)

    # Permutation importance
    print("Computing Permutation Importance...")
    perm_importance = permutation_feature_importance(model, test_dataset)

    return model, importance_maps, perm_importance


def get_experiment_analysis(model, X, Y, experiment_ids, target_experiment_id):
    """
    Return predictions, temporal importance maps, and permutation feature importance
    for a specific experiment ID.

    Parameters:
    - model: trained PyTorch model
    - X: np.array or torch tensor of input data (num_samples, timesteps, features)
    - Y: np.array or torch tensor of true outputs (num_samples, timesteps, outputs)
    - experiment_ids: list or array of experiment IDs corresponding to X/Y
    - target_experiment_id: the experiment ID to analyze

    Returns:
    - predictions: np.array of model predictions for the selected experiment
    - importance_maps: np.array of temporal importance maps for the experiment
    - perm_importance: np.array of permutation feature importance for the experiment
    """
    # Ensure tensors
    X_tensor = (
        torch.tensor(X, dtype=torch.float32)
        if not isinstance(X, torch.Tensor)
        else X.clone().detach()
    )
    Y_tensor = (
        torch.tensor(Y, dtype=torch.float32)
        if not isinstance(Y, torch.Tensor)
        else Y.clone().detach()
    )

    # Select indices for the target experiment
    indices = [
        i for i, exp_id in enumerate(experiment_ids) if exp_id == target_experiment_id
    ]
    if not indices:
        print(f"No data found for experiment ID {target_experiment_id}")
        return None, None, None

    X_exp = X_tensor[indices]
    Y_exp = Y_tensor[indices]

    # Predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_exp).cpu().numpy()

    # Temporal importance maps
    dataset_exp = torch.utils.data.TensorDataset(X_exp, Y_exp)
    importance_maps = model.visualize_cnn_importance(dataset_exp)

    # Permutation feature importance
    perm_importance = permutation_feature_importance(model, dataset_exp)

    return predictions, importance_maps, perm_importance


def plot_temporal_importance_with_real_names(
    importance_maps, df, angle_names=None, max_features=10, selected_angles=None
):
    """
    Plot temporal importance patterns with real feature names from your DataFrame.
    You can pass selected_angles as an int or list of ints to focus on specific angles.
    """

    feature_names = df.columns.tolist()

    if angle_names is None:
        angle_names = [f"Angle {i}" for i in range(importance_maps.shape[0])]

    # Determine which angles to plot
    if selected_angles is None:
        selected_angles = list(
            range(min(importance_maps.shape[0], 5))
        )  # default: first 5
    elif isinstance(selected_angles, int):
        selected_angles = [selected_angles]

    # Create visualization grid
    n_angles = len(selected_angles)
    ncols = min(n_angles + 1, 3)  # +1 for the average
    nrows = (n_angles + 1 + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    # Filter top features if needed
    if len(feature_names) > max_features:
        overall_importance = importance_maps.mean(axis=(0, 1))
        top_feature_indices = np.argsort(overall_importance)[-max_features:][::-1]
        filtered_importance_maps = importance_maps[:, :, top_feature_indices]
        display_feature_names = [feature_names[i] for i in top_feature_indices]
    else:
        filtered_importance_maps = importance_maps
        display_feature_names = feature_names

    # Plot selected angles
    for i, angle_idx in enumerate(selected_angles):
        importance = filtered_importance_maps[angle_idx]
        importance_norm = importance / (importance.max() + 1e-8)
        im = axes[i].imshow(
            importance_norm.T,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        axes[i].set_title(
            f"{angle_names[angle_idx]} - Temporal Importance",
            fontsize=12,
            fontweight="bold",
        )
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel("Feature")
        axes[i].set_yticks(range(len(display_feature_names)))
        axes[i].set_yticklabels(display_feature_names, fontsize=9)

        # Correct colorbar usage
        cbar = plt.colorbar(im, ax=axes[i], pad=0.05)
        cbar.set_label("Normalized Importance", rotation=270, labelpad=15)

    # Average across selected angles
    avg_importance = filtered_importance_maps[selected_angles].mean(axis=0).T
    im = axes[len(selected_angles)].imshow(avg_importance, aspect="auto", cmap="YlOrRd")
    axes[len(selected_angles)].set_title(
        "Average Importance Across Selected Angles", fontsize=12, fontweight="bold"
    )
    axes[len(selected_angles)].set_xlabel("Time Step")
    axes[len(selected_angles)].set_ylabel("Feature")
    axes[len(selected_angles)].set_yticks(range(len(display_feature_names)))
    axes[len(selected_angles)].set_yticklabels(display_feature_names, fontsize=9)

    cbar = plt.colorbar(im, ax=axes[len(selected_angles)], pad=0.05)
    cbar.set_label("Average Importance", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

    return display_feature_names
