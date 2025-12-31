import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_attention_single_angle(experiment_id, angle_idx, model, dfs):
    """
    Visualize features, attention weights (line + heatmap) for a single angle.

    Args:
        experiment_id: ID of the experiment to visualize.
        angle_idx: Which angle's attention to display.
        model: Trained AngleAttentionLSTM model.
        dfs: Pandas DataFrame containing features with 'Experiment_ID' column.
    """
    model.eval()
    
    sample_df = dfs[dfs['Experiment_ID'] == experiment_id].drop(['Experiment_ID'], axis=1)
    X_sample = sample_df.values[np.newaxis, :, :]  
    x_tensor = torch.tensor(X_sample, dtype=torch.float32)

    with torch.no_grad():
        preds, attn_weights = model(x_tensor)

    preds = preds.cpu().numpy()[0]                
    attn_weights = attn_weights.cpu().numpy()[0]  
    
    x_axis = sample_df.index

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)

    for col in sample_df.columns:
        axs[0].plot(x_axis, sample_df[col], marker='.', markersize=4, label=col)
    axs[0].set_ylabel("Feature Value")
    axs[0].set_title(f"Experiment {experiment_id} - Features")
    axs[0].grid(True)
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

    axs[1].plot(x_axis, attn_weights[angle_idx], 'r-', linewidth=2)
    axs[1].set_ylabel("Attention Weight")
    axs[1].set_title(f"Angle {angle_idx} - Attention over Time")
    axs[1].grid(True)

    plt.show()
    x_axis = sample_df.index
    y_axis = np.arange(attn_weights.shape[0])  
    plt.figure(figsize=(14, 6))
    im = plt.imshow(attn_weights, aspect='auto', cmap='magma', 
                    extent=[x_axis[0], x_axis[-1], 0, attn_weights.shape[0]])
    plt.colorbar(im, label="Attention Weight")
    plt.xlabel("Timestep")
    plt.ylabel("Angle Index")
    plt.yticks(y_axis + 0.5, y_axis)  
    plt.title(f"Experiment {experiment_id} - Attention Heatmap (All Angles)")
    plt.show()


def plot_most_accurate_window_lstm(sample_idx, X, Y, model, num_angles, num_features, patch_size, angle_idx=0):
    model.eval()  
    
    _, seq_len, _ = X.shape
    num_patches = seq_len // patch_size
    
    x_input = X[sample_idx].copy()  
    y_true = Y[sample_idx, angle_idx]  
    y_true = np.array(y_true)

    window_errors = []
    for w in range(num_patches):
        x_patch_only = np.zeros_like(x_input)  
        start = w * patch_size
        end = (w + 1) * patch_size
        x_patch_only[start:end, :] = x_input[start:end, :]  
        
        x_tensor = torch.tensor(x_patch_only[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            y_hat = model(x_tensor).cpu().numpy()[0, angle_idx]
        
        if y_true.ndim == 0:
            error = abs(y_true - y_hat)
        else:
            error = np.linalg.norm(y_true - y_hat)
        window_errors.append(error)
    
    window_errors = np.array(window_errors)
    most_accurate_window = np.argmin(window_errors)
    
    plt.figure(figsize=(12, 6))
    for f in range(num_features):
        plt.plot(x_input[:, f], label=f'Feature {f}')
    
    plt.axvspan(most_accurate_window*patch_size,
                (most_accurate_window+1)*patch_size-1,
                color='green', alpha=0.3, label='Most Accurate Window')
    
    error_curve = np.repeat(window_errors, patch_size)
    ax2 = plt.gca().twinx()
    ax2.plot(range(len(error_curve)), error_curve, color='black', linestyle='--',
             linewidth=2, label='Prediction Error per Patch')
    ax2.set_ylabel("Error (L2 norm if vector)")
    
    plt.xlabel("Timestep")
    plt.title(f"Sample {sample_idx}, Angle {angle_idx} - Most Accurate Window & Error Curve")
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()
    
    x_tensor = torch.tensor(x_input[np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy()[0]  
    
    plt.figure(figsize=(12, 6))
    for out_idx in range(Y.shape[2]):
        plt.plot(range(num_angles), Y[sample_idx, :, out_idx], label=f'True Output {out_idx}')
        plt.plot(range(num_angles), preds[:, out_idx], label=f'Predicted Output {out_idx}', linestyle='--', marker='x')
    plt.xlabel("Angle index")
    plt.ylabel("Target value(s)")
    plt.title(f"Predictions for Sample {sample_idx} across all angles")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
    
def plot_most_accurate_window_angle_lstm(sample_idx, X, Y, model, num_angles, num_features, patch_size, angle_idx=0):

    model.eval()  
    
    _, seq_len, _ = X.shape
    num_patches = seq_len // patch_size
    
    x_input = X[sample_idx].copy()  
    y_true = Y[sample_idx, angle_idx]  
    y_true = np.array(y_true)

    window_errors = []
    for w in range(num_patches):
        x_patch_only = np.zeros_like(x_input)  
        start = w * patch_size
        end = (w + 1) * patch_size
        x_patch_only[start:end, :] = x_input[start:end, :]  
        
        x_tensor = torch.tensor(x_patch_only[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            y_hat, _ = model(x_tensor)  
            y_hat = y_hat.cpu().numpy()[0, angle_idx]  
        
        if y_true.ndim == 0:
            error = abs(y_true - y_hat)
        else:
            error = np.linalg.norm(y_true - y_hat)
        window_errors.append(error)
    
    window_errors = np.array(window_errors)
    most_accurate_window = np.argmin(window_errors)
    
    plt.figure(figsize=(12, 6))
    for f in range(num_features):
        plt.plot(x_input[:, f], label=f'Feature {f}')
    
    plt.axvspan(most_accurate_window*patch_size,
                (most_accurate_window+1)*patch_size-1,
                color='green', alpha=0.3, label='Most Accurate Window')
    
    error_curve = np.repeat(window_errors, patch_size)
    ax2 = plt.gca().twinx()
    ax2.plot(range(len(error_curve)), error_curve, color='black', linestyle='--',
             linewidth=2, label='Prediction Error per Patch')
    ax2.set_ylabel("Error (L2 norm if vector)")
    
    plt.xlabel("Timestep")
    plt.title(f"Sample {sample_idx}, Angle {angle_idx} - Most Accurate Window & Error Curve")
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()
    
    x_tensor = torch.tensor(x_input[np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        preds, attn_weights = model(x_tensor)  
        preds = preds.cpu().numpy()[0]  
    
    plt.figure(figsize=(12, 6))
    for out_idx in range(Y.shape[2]):
        plt.plot(range(num_angles), Y[sample_idx, :, out_idx], label=f'True Output {out_idx}')
        plt.plot(range(num_angles), preds[:, out_idx], label=f'Predicted Output {out_idx}', linestyle='--', marker='x')
    plt.xlabel("Angle index")
    plt.ylabel("Target value(s)")
    plt.title(f"Predictions for Sample {sample_idx} across all angles")
    plt.legend()
    plt.grid(True)
    plt.show()
