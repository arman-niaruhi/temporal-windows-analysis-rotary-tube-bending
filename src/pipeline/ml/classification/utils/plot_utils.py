import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import numpy as np
from scipy.ndimage import binary_opening, binary_closing

def plot_predictions_vs_true_annot(model, dataset, sensor_df, feature_cols, test_exps, morphing=False, device="cpu"):
    """
    Multi-panel plot:
    - Top subplot: sensor data (features vs. time)
    - Middle subplot: predictions as colored spans
    - Bottom subplot: true labels as colored spans
    Each label gets its own color, shown in a shared legend.
    """
    model.eval()

    # --- Select a random experiment ---
    exp_id = random.choice(test_exps)
    exp_data = sensor_df[sensor_df["Experiment_ID"] == exp_id]

    # --- Features ---
    X = torch.tensor(exp_data[feature_cols].values, dtype=torch.float32).to(device)
    X = X.unsqueeze(0)  # [1, seq_len, num_features]

    # --- True labels ---
    y_true = exp_data["Label"].values

    # --- Model predictions ---
    with torch.no_grad():
        outputs = model(X)
        y_pred = torch.argmax(outputs, dim=-1).squeeze(0).cpu().numpy()
        
    # Size of neighborhood to consider
    kernel_size = 5  # adjust: larger = more aggressive smoothing

    # Process each unique label
    smoothed = np.zeros_like(y_pred)
    for label in np.unique(y_pred):
        mask = y_pred == label
        # Apply morphological closing (fills small gaps)
        mask = binary_closing(mask, structure=np.ones(kernel_size))
        # Apply morphological opening (removes small noise)
        mask = binary_opening(mask, structure=np.ones(kernel_size))
        smoothed[mask] = label
        
    y_pred = smoothed
    
    # --- Map indices to label names ---
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    y_true_names = [idx_to_label[label] if isinstance(label, int) else label for label in y_true]
    y_pred_names = [idx_to_label[p] for p in y_pred]

    # --- Plotting ---
    timestamps = exp_data.index.astype(float)
    fig, axes = plt.subplots(
        4, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1, 1, 0.5]}
    )
    ax_data, ax_pred, ax_true, ax_leg = axes

    # --- Top subplot: Sensor data ---
    for col in feature_cols:
        ax_data.plot(timestamps, exp_data[col].values, label=col)
    ax_data.set_title(f"Experiment {exp_id} — Sensor Data & Predictions")
    ax_data.set_ylabel("Sensor values")

    # --- Assign unique colors to each label ---
    unique_labels = sorted(set(y_pred_names) | set(y_true_names))
    cmap = plt.get_cmap("tab10")
    label_to_color = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}

    # --- Middle subplot: Predictions ---
    for i in range(len(timestamps) - 1):
        t_start, t_end = timestamps[i], timestamps[i+1]
        pred_label = y_pred_names[i]
        ax_pred.axvspan(t_start, t_end, color=label_to_color[pred_label], alpha=0.3)
    ax_pred.set_ylabel("Prediction")
    ax_pred.set_yticks([])

    # --- Bottom subplot: True labels ---
    for i in range(len(timestamps) - 1):
        t_start, t_end = timestamps[i], timestamps[i+1]
        true_label = y_true_names[i]
        ax_true.axvspan(t_start, t_end, color=label_to_color[true_label], alpha=0.3)
    ax_true.set_ylabel("True Label")
    ax_true.set_xlabel("Time")
    ax_true.set_yticks([])

    # --- Legend subplot ---
    patches = [mpatches.Patch(color=color, alpha=0.3, label=label)
               for label, color in label_to_color.items()]
    ax_leg.axis("off")
    ax_leg.legend(handles=patches, loc="center", ncol=5, frameon=False)

    plt.tight_layout()
    plt.show()
