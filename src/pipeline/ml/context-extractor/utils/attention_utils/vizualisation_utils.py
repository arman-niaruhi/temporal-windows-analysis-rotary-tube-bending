import matplotlib.pyplot as plt
import numpy as np
import torch

def compute_window_errors(x_input, y_true, model, patch_size, angle_idx=0):
    """Compute prediction error per patch and return most accurate window."""
    seq_len, num_features = x_input.shape
    num_patches = seq_len // patch_size
    window_errors = []

    for w in range(num_patches):
        x_patch_only = np.zeros_like(x_input)
        start, end = w*patch_size, (w+1)*patch_size
        x_patch_only[start:end, :] = x_input[start:end, :]
        x_tensor = torch.tensor(x_patch_only[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            y_hat = model(x_tensor)
            if isinstance(y_hat, tuple):  # model returns (preds, attn)
                y_hat = y_hat[0]
            y_hat = y_hat.cpu().numpy()[0, angle_idx]

        error = np.linalg.norm(y_true - y_hat) if y_true.ndim else abs(y_true - y_hat)
        window_errors.append(error)

    window_errors = np.array(window_errors)
    most_accurate_window = np.argmin(window_errors)
    return window_errors, most_accurate_window

def plot_features_with_error(x_input, window_errors, patch_size, most_accurate_window, num_features, title=""):
    """Plot features, highlight most accurate window, and overlay error curve."""
    plt.figure(figsize=(12, 6))
    for f in range(num_features):
        plt.plot(x_input[:, f], label=f'Feature {f}')

    # Highlight most accurate window
    plt.axvspan(most_accurate_window*patch_size,
                (most_accurate_window+1)*patch_size-1,
                color='green', alpha=0.3, label='Most Accurate Window')

    # Overlay error curve
    error_curve = np.repeat(window_errors, patch_size)
    ax2 = plt.gca().twinx()
    ax2.plot(range(len(error_curve)), error_curve, 'k--', linewidth=2, label='Prediction Error')
    ax2.set_ylabel("Error")
    plt.xlabel("Timestep")
    plt.title(title)
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

def plot_true_vs_pred(Y_sample, preds, title="Predictions vs True"):
    """Plot predictions vs true outputs across all angles."""
    num_angles, output_size = preds.shape
    plt.figure(figsize=(12, 6))
    for out_idx in range(output_size):
        plt.plot(range(num_angles), Y_sample[:, out_idx], label=f'True {out_idx}')
        plt.plot(range(num_angles), preds[:, out_idx], '--x', label=f'Pred {out_idx}')
    plt.xlabel("Angle index")
    plt.ylabel("Output value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def visualize_attention(experiment_id, angle_idx, model, dfs):
    """Plot features and attention weights for a single experiment/angle."""
    model.eval()
    sample_df = dfs[dfs['Experiment_ID']==experiment_id].drop('Experiment_ID', axis=1)
    X_sample = sample_df.values[np.newaxis, :, :]
    x_tensor = torch.tensor(X_sample, dtype=torch.float32)

    with torch.no_grad():
        preds, attn_weights = model(x_tensor)
    preds, attn_weights = preds.cpu().numpy()[0], attn_weights.cpu().numpy()[0]

    x_axis = sample_df.index

    # Features plot
    plt.figure(figsize=(14, 4))
    for col in sample_df.columns:
        plt.plot(x_axis, sample_df[col], marker='.', markersize=4, label=col)
    plt.title(f"Experiment {experiment_id} - Features")
    plt.ylabel("Feature Value")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=8)
    plt.show()

    # Attention line plot
    plt.figure(figsize=(14, 3))
    plt.plot(x_axis, attn_weights[angle_idx], 'r-', linewidth=2)
    plt.title(f"Angle {angle_idx} - Attention over Time")
    plt.ylabel("Attention Weight")
    plt.grid(True)
    plt.show()

    # Attention heatmap
    plt.figure(figsize=(14, 5))
    plt.imshow(attn_weights, aspect='auto', cmap='magma', extent=[x_axis[0], x_axis[-1], 0, attn_weights.shape[0]])
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Timestep")
    plt.ylabel("Angle Index")
    plt.yticks(np.arange(attn_weights.shape[0])+0.5, np.arange(attn_weights.shape[0]))
    plt.title(f"Experiment {experiment_id} - Attention Heatmap (All Angles)")
    plt.show()
