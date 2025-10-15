import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# -----------------------------
# Sample Retrieval
# -----------------------------
def get_sample_by_index(X_flat, num_angles, sample_idx=None, local_idx=None, angle_idx=None):
    """
    Retrieve the correct sequence and angle given either:
      - a global flattened index (sample_idx)
      - or a local sample index (local_idx) + angle index (angle_idx)
    """
    if sample_idx is not None:
        # global index
        X_seq = X_flat[sample_idx]
        angle_idx = 0 if num_angles <= 1 else sample_idx % num_angles
        angle_value = 0.0 if num_angles <= 1 else angle_idx / (num_angles - 1)
    elif local_idx is not None and angle_idx is not None:
        # compute global index from local sample + angle
        flat_idx = local_idx * max(1, num_angles) + angle_idx
        if flat_idx >= len(X_flat):
            raise IndexError(f"Computed flat_idx={flat_idx} exceeds X_flat length={len(X_flat)}")
        X_seq = X_flat[flat_idx]
        angle_value = 0.0 if num_angles <= 1 else angle_idx / (num_angles - 1)
    else:
        raise ValueError("Provide either sample_idx or (local_idx + angle_idx).")
    
    return X_seq, angle_value, angle_idx

# -----------------------------
# Importance Functions
# -----------------------------
def compute_ig_importance(model, X_flat, num_angles, sample_idx=None, local_idx=None, angle_idx=None, target_feature_idx=0):
    model.eval()
    X_seq, angle_value, _ = get_sample_by_index(X_flat, num_angles, sample_idx, local_idx, angle_idx)
    ig = IntegratedGradients(model)
    X_seq = X_seq.unsqueeze(0).requires_grad_(True)
    angle_tensor = torch.tensor([[angle_value]], dtype=torch.float32)
    attr_tuple, _ = ig.attribute((X_seq, angle_tensor), target=target_feature_idx, n_steps=50, return_convergence_delta=True)
    importance = attr_tuple[0].detach().numpy().squeeze()
    return np.mean(np.abs(importance), axis=1)

def compute_grad_input_importance(model, X_flat, num_angles, sample_idx=None, local_idx=None, angle_idx=None, target_feature_idx=0):
    model.eval()
    X_seq, angle_value, _ = get_sample_by_index(X_flat, num_angles, sample_idx, local_idx, angle_idx)
    X_seq = X_seq.unsqueeze(0).requires_grad_(True)
    angle_tensor = torch.tensor([[angle_value]], dtype=torch.float32)
    output = model(X_seq, angle_tensor)
    loss = output[0, target_feature_idx]
    loss.backward()
    grad = X_seq.grad.detach().numpy().squeeze()
    importance = X_seq.detach().numpy().squeeze() * grad
    return np.mean(np.abs(importance), axis=1)

def compute_saliency(model, X_flat, num_angles, sample_idx=None, local_idx=None, angle_idx=None, target_feature_idx=0):
    model.eval()
    X_seq, angle_value, _ = get_sample_by_index(X_flat, num_angles, sample_idx, local_idx, angle_idx)
    X_seq = X_seq.unsqueeze(0).requires_grad_(True)
    angle_tensor = torch.tensor([[angle_value]], dtype=torch.float32)
    output = model(X_seq, angle_tensor)
    loss = output[0, target_feature_idx]
    loss.backward()
    saliency = X_seq.grad.detach().numpy().squeeze()
    return np.max(np.abs(saliency), axis=1)

def compute_occlusion_importance(model, X_flat, num_angles, sample_idx=None, local_idx=None, angle_idx=None, target_feature_idx=0, window=1):
    model.eval()
    X_seq, angle_value, _ = get_sample_by_index(X_flat, num_angles, sample_idx, local_idx, angle_idx)
    baseline = X_seq.clone()
    original_output = model(X_seq.unsqueeze(0), torch.tensor([[angle_value]])).detach()[0, target_feature_idx]
    seq_len = X_seq.shape[0]
    importance = np.zeros(seq_len)
    for t in range(seq_len):
        X_perturbed = baseline.clone()
        start = max(0, t - window // 2)
        end = min(seq_len, t + window // 2 + 1)
        X_perturbed[start:end] = 0
        pert_output = model(X_perturbed.unsqueeze(0), torch.tensor([[angle_value]])).detach()[0, target_feature_idx]
        importance[t] = abs(original_output - pert_output)
    return importance

def compute_attention_importance_from_seq(model, X_seq, angle_value):
    model.eval()
    with torch.no_grad():
        angle_tensor = torch.tensor([[angle_value]], dtype=torch.float32)
        _, attn_weights = model(X_seq.unsqueeze(0), angle_tensor, return_attention=True)
        return attn_weights.squeeze(0).detach().numpy()

# -----------------------------
# Plotting Function
# -----------------------------
def plot_all_importances_for_sample_safe(model, sample_idx_local, angle_indices, X_flat, num_angles, sensors_df, target_feature_idx=0):
    """
    Plot feature importances for a specific local sample and specific angles.
    Uses global indices internally.
    """
    methods = ['Integrated Gradients', 'Grad × Input', 'Saliency', 'Occlusion', 'Attention']
    all_importances = {m: [] for m in methods}
    experiment_idx = sensors_df['Experiment_ID'].unique()[sample_idx_local]
    sensors_df_selected = sensors_df[sensors_df['Experiment_ID'] == experiment_idx].copy()
    sensors_df_selected = sensors_df_selected.drop(columns=['Experiment_ID'], errors='ignore')
    seq_len = len(sensors_df_selected)
    if seq_len == 0:
        raise ValueError(f"sensors_df for sample_idx_local={sample_idx_local} is empty.")
    x_axis = np.arange(seq_len)
    base_sample_idx = sample_idx_local

    for angle_idx in angle_indices:
        if angle_idx >= max(1, num_angles):
            print(f"Skipping angle_idx={angle_idx} because it exceeds num_angles={num_angles}")
            continue
        flat_idx = base_sample_idx * max(1, num_angles) + angle_idx
        if flat_idx >= len(X_flat):
            print(f"Skipping flat_idx={flat_idx} because X_flat has length {len(X_flat)}")
            continue
        X_seq = X_flat[flat_idx]
        angle_value = 0.0 if num_angles <= 1 else angle_idx / (num_angles - 1)
        try:
            all_importances['Integrated Gradients'].append(
                compute_ig_importance(model, X_flat, num_angles, sample_idx=flat_idx, target_feature_idx=target_feature_idx)
            )
            all_importances['Grad x Input'].append(
                compute_grad_input_importance(model, X_flat, num_angles, sample_idx=flat_idx, target_feature_idx=target_feature_idx)
            )
            all_importances['Saliency'].append(
                compute_saliency(model, X_flat, num_angles, sample_idx=flat_idx, target_feature_idx=target_feature_idx)
            )
            all_importances['Occlusion'].append(
                compute_occlusion_importance(model, X_flat, num_angles, sample_idx=flat_idx, target_feature_idx=target_feature_idx)
            )
            all_importances['Attention'].append(
                compute_attention_importance_from_seq(model, X_seq, angle_value)
            )
        except Exception as e:
            print(f"Skipping angle_idx={angle_idx} due to error: {e}")
            continue

    # Plot results
    for method in methods:
        data_list = all_importances[method]
        if len(data_list) == 0:
            print(f"No importance data computed for {method}, skipping plot.")
            continue
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 2]})
        axs[0].plot(x_axis, sensors_df_selected.values)
        axs[0].set_title("Raw Sensor Data")
        axs[0].set_xlabel("Time step")
        axs[0].set_ylabel("Sensor values")
        data = np.array(data_list)
        data = np.abs(data)
        data = np.squeeze(data)
        if data.ndim == 1:
            data = data[None, :]
        elif data.ndim > 2:
            data = np.mean(data, axis=tuple(range(2, data.ndim)))
        axs[1].imshow(data, aspect='auto', origin='lower', cmap='viridis',
                      extent=[x_axis[0], x_axis[-1], 0, len(data) - 1])
        axs[1].set_title(f"{method} Importance Across Angles (Local idx: {sample_idx_local})")
        axs[1].set_xlabel("Time step")
        axs[1].set_ylabel("Angle index")
        fig.colorbar(axs[1].images[0], ax=axs[1], label='Importance')
        plt.tight_layout()
        plt.show()
