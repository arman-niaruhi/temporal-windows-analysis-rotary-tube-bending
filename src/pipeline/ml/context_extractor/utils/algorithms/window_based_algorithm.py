import torch
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)



def window_based_importance(
    model,
    train_loader,
    n_angle: int,
    occluded_window_size: int = 10,
    stride: int = 5,
    device=None,
):
    """
    Compute mean window-based importance across ALL samples in the dataloader.

    Returns:
        importance_df: DataFrame of mean importance per window
        mean_importance: np.ndarray of shape (n_windows,)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_importance = []
    n_windows = None

    with torch.no_grad():
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            batch_size, T, F = X_batch.shape

            for b in range(batch_size):
                x = X_batch[b:b + 1]

                original_pred, _ = model(x)
                original_pred = original_pred.cpu().numpy()

                importance_vals = []

                for start in range(0, T - occluded_window_size + 1, stride):
                    x_occluded = x.clone()
                    x_occluded[:, start:start + occluded_window_size, :] = 0.0

                    occluded_pred, _ = model(x_occluded)
                    occluded_pred = occluded_pred.cpu().numpy()

                    delta = np.mean(
                        np.abs(
                            original_pred[:, n_angle:n_angle + 1, :]
                            - occluded_pred[:, n_angle:n_angle + 1, :]
                        )
                    )
                    importance_vals.append(delta)

                all_importance.append(importance_vals)

                if n_windows is None:
                    n_windows = len(importance_vals)

    all_importance = np.asarray(all_importance)          # (N_samples, n_windows)
    mean_importance = all_importance.mean(axis=0)        # (n_windows,)

    importance_df = pd.DataFrame(
        [mean_importance],
        columns=[f"window_{i}" for i in range(n_windows)]
    )

    return importance_df, mean_importance


def save_window_importance_results(all_importance_data, output_dir):
    """
    Save window-based importance results for multiple angles to a CSV file.

    Args:
        all_importance_data (list of tuples): Each tuple is (n_angle, importance_df, mean_importance)
        output_path (str): Path to save the CSV file
    """
    combined_data = []

    for n_angle, importance_df, mean_importance in all_importance_data:
        df = importance_df.copy()
        df.insert(0, "angle", n_angle)  # Add a column for the angle
        combined_data.append(df)

    all_df = pd.concat(combined_data, ignore_index=True)
    # Ensure the directory exists
    output_path = output_dir / "window_importance_all_angles.csv"
    all_df.to_csv(output_path, index=False)
    
    