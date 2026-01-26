from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from tqdm import tqdm


def _save_timestep_heatmap(
    sensitivity: np.ndarray,
    saving_dir: Path,
    filename: str,
    title: str,
    annot_timesteps: Optional[list[int]] = None,
    mandrel_extraction_annot_timesteps: Optional[list[int]] = None,
    sensor_data: Optional[np.ndarray] = None,
    sensor_names: Optional[list[str]] = None,
) -> Path:
    if sensor_data is not None:
        seq_len = sensor_data.shape[0]
    else:
        seq_len = sensitivity.shape[1]
    ref_max = _reference_max(annot_timesteps, mandrel_extraction_annot_timesteps)
    scaled_annot = _scale_timesteps(annot_timesteps, seq_len, ref_max)
    scaled_mandrel = _scale_span(mandrel_extraction_annot_timesteps, seq_len, ref_max)

    if sensor_data is not None:
        fig = plt.figure(figsize=(25, 12), facecolor="white")
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.25)
        ax_sensor = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1], sharex=ax_sensor)
        handles, labels = _plot_sensor_subplot(
            ax_sensor,
            sensor_data,
            sensor_names,
            scaled_annot,
            scaled_mandrel,
        )
        legend = ax_sensor.legend(
            handles,
            labels,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=10,
            framealpha=0.95,
            edgecolor="#cccccc",
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_linewidth(1.2)
    else:
        fig, ax = plt.subplots(figsize=(14, 6))

    n_angles = sensitivity.shape[0]
    im = ax.imshow(
        sensitivity,
        aspect="auto",
        cmap="magma",
        interpolation="bilinear",
        extent=[0, seq_len - 1, n_angles - 1, 0],
    )
    ax.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Angle Index", fontsize=9, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_yticks(np.arange(n_angles))
    ax.set_yticklabels(
        [f"{i + 1}" for i in reversed(range(n_angles))],
        fontsize=5,
    )

    if sensor_data is not None:
        ax.set_xlim(0, sensor_data.shape[0] - 1)

    if scaled_annot:
        for ts in scaled_annot:
            ax.axvline(ts, color="white", linestyle="--", linewidth=1, alpha=0.6)

    if scaled_mandrel and len(scaled_mandrel) == 2:
        ax.axvspan(
            scaled_mandrel[0],
            scaled_mandrel[1],
            color="white",
            alpha=0.15,
            linewidth=0,
        )

    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Sensitivity (delta MSE)", fontsize=11, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(1.2)

    plt.tight_layout()
    if sensor_data is not None:
        pos_main = ax_sensor.get_position()
        pos_heat = ax.get_position()
        ax.set_position([pos_heat.x0, pos_heat.y0, pos_main.width, pos_heat.height])
        cbar.ax.set_position(
            [pos_main.x0 + pos_main.width + 0.02, pos_heat.y0, 0.015, pos_heat.height]
        )
    out_path = saving_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_window_heatmap(
    sensitivity: np.ndarray,
    saving_dir: Path,
    filename: str,
    title: str,
    window_starts: list[int],
    window_size: int,
    annot_timesteps: Optional[list[int]] = None,
    mandrel_extraction_annot_timesteps: Optional[list[int]] = None,
    sensor_data: Optional[np.ndarray] = None,
    sensor_names: Optional[list[str]] = None,
) -> Path:
    seq_len = (
        sensor_data.shape[0]
        if sensor_data is not None
        else (window_starts[-1] + window_size if window_starts else sensitivity.shape[1])
    )
    ref_max = _reference_max(annot_timesteps, mandrel_extraction_annot_timesteps)
    scaled_annot = _scale_timesteps(annot_timesteps, seq_len, ref_max)
    scaled_mandrel = _scale_span(mandrel_extraction_annot_timesteps, seq_len, ref_max)

    if sensor_data is not None:
        fig = plt.figure(figsize=(25, 12), facecolor="white")
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.25)
        ax_sensor = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1], sharex=ax_sensor)
        handles, labels = _plot_sensor_subplot(
            ax_sensor,
            sensor_data,
            sensor_names,
            scaled_annot,
            scaled_mandrel,
        )
        legend = ax_sensor.legend(
            handles,
            labels,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=10,
            framealpha=0.95,
            edgecolor="#cccccc",
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_linewidth(1.2)
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
    n_angles, n_windows = sensitivity.shape
    if n_windows == 0:
        raise ValueError("No windows available to plot.")

    x_edges = np.array(window_starts + [window_starts[-1] + window_size], dtype=float)
    y_edges = np.arange(0, n_angles + 1, dtype=float)

    pcm = ax.pcolormesh(x_edges, y_edges, sensitivity, shading="auto", cmap="magma")
    ax.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Angle Index", fontsize=9, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_yticks(np.arange(n_angles))
    ax.set_yticklabels(
        [f"{i + 1}" for i in reversed(range(n_angles))],
        fontsize=5,
    )
    ax.invert_yaxis()

    if scaled_annot:
        for ts in scaled_annot:
            ax.axvline(ts, color="white", linestyle="--", linewidth=1, alpha=0.6)

    if scaled_mandrel and len(scaled_mandrel) == 2:
        ax.axvspan(
            scaled_mandrel[0],
            scaled_mandrel[1],
            color="white",
            alpha=0.15,
            linewidth=0,
        )

    if sensor_data is not None:
        ax.set_xlim(0, sensor_data.shape[0] - 1)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    cbar = plt.colorbar(pcm, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Sensitivity (delta MSE)", fontsize=11, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(1.2)

    plt.tight_layout()
    if sensor_data is not None:
        pos_main = ax_sensor.get_position()
        pos_heat = ax.get_position()
        ax.set_position([pos_heat.x0, pos_heat.y0, pos_main.width, pos_heat.height])
        cbar.ax.set_position(
            [pos_main.x0 + pos_main.width + 0.02, pos_heat.y0, 0.015, pos_heat.height]
        )
    out_path = saving_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_sensor_subplot(
    ax: plt.Axes,
    sensor_data: np.ndarray,
    sensor_names: Optional[list[str]],
    annot_timesteps: Optional[list[int]],
    mandrel_extraction_annot_timesteps: Optional[list[int]],
) -> tuple[list[plt.Line2D], list[str]]:
    time_steps = np.arange(sensor_data.shape[0])
    names = sensor_names or [f"f{i}" for i in range(sensor_data.shape[1])]
    cleaned_names = [name.replace("_mean", "") for name in names]

    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(names)))

    handles = []
    labels = []
    for i, (name, color) in enumerate(zip(cleaned_names, colors)):
        line = ax.plot(
            time_steps,
            sensor_data[:, i],
            color=color,
            linewidth=2.5,
            alpha=0.8,
            label=name,
            marker="o",
            markersize=3,
            markevery=max(1, sensor_data.shape[0] // 20),
        )[0]
        handles.append(line)
        labels.append(name)

    if annot_timesteps:
        for ts in annot_timesteps:
            ax.axvline(ts, color="black", linestyle="--", linewidth=1, alpha=0.5)

    if annot_timesteps and len(annot_timesteps) == 4:
        annot_labels = [
            "Start-Clamping",
            "Start-Bending",
            "Start-Declamping",
            "End-Declamping",
        ]
        y_max = float(np.max(sensor_data)) if sensor_data.size else 1.0
        for ts, label in zip(annot_timesteps, annot_labels):
            ax.annotate(
                label,
                xy=(ts, y_max),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.8),
            )

    if mandrel_extraction_annot_timesteps and len(mandrel_extraction_annot_timesteps) == 2:
        ax.axvspan(
            mandrel_extraction_annot_timesteps[0],
            mandrel_extraction_annot_timesteps[1],
            color="blue",
            alpha=0.12,
            linewidth=0,
        )

    ax.set_xlim(0, sensor_data.shape[0] - 1)
    if sensor_data.size:
        y_min = float(np.min(sensor_data))
        y_max = float(np.max(sensor_data))
        padding = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
        ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Feature Value", fontsize=12, fontweight="bold", labelpad=10)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.8, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.set_facecolor("#f9f9f9")
    ax.set_title("Sensor Data Over Time", fontsize=14, fontweight="bold", pad=15)
    return handles, labels


def _scale_timesteps(
    timesteps: Optional[list[int]],
    seq_len: int,
    reference_max: Optional[int] = None,
) -> Optional[list[int]]:
    if not timesteps:
        return None
    max_ts = reference_max if reference_max is not None else max(timesteps)
    if max_ts <= seq_len - 1 or max_ts == 0:
        return timesteps
    scale = (seq_len - 1) / max_ts
    return [int(round(ts * scale)) for ts in timesteps]


def _scale_span(
    span: Optional[list[int]],
    seq_len: int,
    reference_max: Optional[int] = None,
) -> Optional[list[int]]:
    if not span or len(span) != 2:
        return None
    scaled = _scale_timesteps(span, seq_len, reference_max)
    if not scaled:
        return None
    return [max(0, min(seq_len - 1, scaled[0])), max(0, min(seq_len - 1, scaled[1]))]


def _reference_max(
    annot_timesteps: Optional[list[int]],
    mandrel_extraction_annot_timesteps: Optional[list[int]],
) -> Optional[int]:
    values: list[int] = []
    if annot_timesteps:
        values.extend(annot_timesteps)
    if mandrel_extraction_annot_timesteps:
        values.extend(mandrel_extraction_annot_timesteps)
    return max(values) if values else None


def compute_timestep_ablation_sensitivity(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    saving_dir: Path,
    annot_timesteps: Optional[list[int]] = None,
    mandrel_extraction_annot_timesteps: Optional[list[int]] = None,
    sensor_data: Optional[np.ndarray] = None,
    sensor_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Compute per-timestep sensitivity by zeroing all features at each timestep.

    Returns:
        DataFrame indexed by angle index with columns per timestep.
        Path to saved heatmap.
    """
    model.eval()
    saving_dir.mkdir(parents=True, exist_ok=True)

    baseline_loss_sum: Optional[torch.Tensor] = None
    n_batches = 0
    seq_len = None

    with torch.no_grad():
        for Xb, Yb, springback, experiment_config in val_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            springback = springback.to(device)
            experiment_config = experiment_config.to(device)

            pred, _ = model(Xb, springback, experiment_config)
            mse_per_angle = (pred - Yb).pow(2).mean(dim=(0, 2))
            baseline_loss_sum = (
                mse_per_angle if baseline_loss_sum is None else baseline_loss_sum + mse_per_angle
            )
            n_batches += 1
            if seq_len is None:
                seq_len = Xb.shape[1]

    if n_batches == 0 or seq_len is None:
        raise ValueError("Validation loader is empty; cannot compute timestep sensitivity.")

    baseline_loss = (baseline_loss_sum / n_batches).detach().cpu().numpy()
    n_angles = baseline_loss.shape[0]
    sensitivity = np.zeros((n_angles, seq_len), dtype=np.float32)

    for t in tqdm(range(seq_len), desc="Timestep ablation"):
        ablated_loss_sum = None
        n_batches = 0
        with torch.no_grad():
            for Xb, Yb, springback, experiment_config in val_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                springback = springback.to(device)
                experiment_config = experiment_config.to(device)

                Xb_ablated = Xb.clone()
                Xb_ablated[:, t, :] = 0

                pred, _ = model(Xb_ablated, springback, experiment_config)
                mse_per_angle = (pred - Yb).pow(2).mean(dim=(0, 2))
                ablated_loss_sum = (
                    mse_per_angle if ablated_loss_sum is None else ablated_loss_sum + mse_per_angle
                )
                n_batches += 1

        ablated_loss = (ablated_loss_sum / n_batches).detach().cpu().numpy()
        sensitivity[:, t] = ablated_loss - baseline_loss

    df = pd.DataFrame(
        sensitivity,
        index=[f"angle_{i}" for i in range(n_angles)],
        columns=[f"t{t}" for t in range(seq_len)],
    )
    csv_path = saving_dir / "timestep_ablation_sensitivity.csv"
    df.to_csv(csv_path)

    heatmap_path = _save_timestep_heatmap(
        sensitivity,
        saving_dir,
        "timestep_ablation_sensitivity.png",
        "Timestep Sensitivity per Angle (Ablation)",
        annot_timesteps,
        mandrel_extraction_annot_timesteps,
        sensor_data,
        sensor_names,
    )
    return df, heatmap_path


def compute_timestep_ig_sensitivity(
    model: torch.nn.Module,
    X_sample: torch.Tensor,
    springback_sample: Optional[torch.Tensor],
    experiment_config: Optional[torch.Tensor],
    saving_dir: Path,
    annot_timesteps: Optional[list[int]] = None,
    mandrel_extraction_annot_timesteps: Optional[list[int]] = None,
    target_feature_idx: Optional[int] = None,
    n_steps: int = 50,
    sensor_data: Optional[np.ndarray] = None,
    sensor_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Compute per-timestep sensitivity using Integrated Gradients for each angle index.
    Aggregates attribution across input features per timestep.
    """
    model.eval()
    saving_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    X_sample = X_sample.to(device)
    if X_sample.dim() == 2:
        X_sample = X_sample.unsqueeze(0)

    if springback_sample is not None:
        springback_sample = springback_sample.to(device)
    if experiment_config is not None:
        experiment_config = experiment_config.to(device)

    with torch.no_grad():
        pred, _ = model(X_sample, springback_sample, experiment_config)

    n_angles = pred.shape[1]
    seq_len = X_sample.shape[1]
    sensitivity = np.zeros((n_angles, seq_len), dtype=np.float32)

    for angle_idx in range(n_angles):
        def forward_for_ig(x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]

            springback_expanded = None
            if springback_sample is not None:
                sb = springback_sample
                if sb.dim() == 1:
                    sb = sb.unsqueeze(0)
                springback_expanded = sb.expand(batch_size, -1)

            config_expanded = None
            if experiment_config is not None:
                cfg = experiment_config
                if cfg.dim() == 1:
                    cfg = cfg.unsqueeze(0)
                config_expanded = cfg.expand(batch_size, -1)

            pred_local, _ = model(x, springback_expanded, config_expanded)
            if target_feature_idx is None:
                return pred_local[:, angle_idx, :].sum(dim=-1)
            return pred_local[:, angle_idx, target_feature_idx]

        ig = IntegratedGradients(forward_for_ig)
        attributions, _ = ig.attribute(X_sample, n_steps=n_steps, return_convergence_delta=True)
        attrib = attributions.squeeze(0).detach().cpu().numpy()
        sensitivity[angle_idx] = np.abs(attrib).sum(axis=-1)

    df = pd.DataFrame(
        sensitivity,
        index=[f"angle_{i}" for i in range(n_angles)],
        columns=[f"t{t}" for t in range(seq_len)],
    )
    csv_path = saving_dir / "timestep_ig_sensitivity.csv"
    df.to_csv(csv_path)

    heatmap_path = _save_timestep_heatmap(
        sensitivity,
        saving_dir,
        "timestep_ig_sensitivity.png",
        "Timestep Sensitivity per Angle (Integrated Gradients)",
        annot_timesteps,
        mandrel_extraction_annot_timesteps,
        sensor_data,
        sensor_names,
    )
    return df, heatmap_path


def compute_timestep_window_occlusion_sensitivity(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    saving_dir: Path,
    occluded_window_size: int,
    stride: int,
    annot_timesteps: Optional[list[int]] = None,
    mandrel_extraction_annot_timesteps: Optional[list[int]] = None,
    sensor_data: Optional[np.ndarray] = None,
    sensor_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Compute per-window sensitivity by zeroing a sliding window of timesteps.
    """
    model.eval()
    saving_dir.mkdir(parents=True, exist_ok=True)

    baseline_loss_sum: Optional[torch.Tensor] = None
    n_batches = 0
    seq_len = None

    with torch.no_grad():
        for Xb, Yb, springback, experiment_config in val_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            springback = springback.to(device)
            experiment_config = experiment_config.to(device)

            pred, _ = model(Xb, springback, experiment_config)
            mse_per_angle = (pred - Yb).pow(2).mean(dim=(0, 2))
            baseline_loss_sum = (
                mse_per_angle if baseline_loss_sum is None else baseline_loss_sum + mse_per_angle
            )
            n_batches += 1
            if seq_len is None:
                seq_len = Xb.shape[1]

    if n_batches == 0 or seq_len is None:
        raise ValueError("Validation loader is empty; cannot compute window sensitivity.")

    baseline_loss = (baseline_loss_sum / n_batches).detach().cpu().numpy()
    n_angles = baseline_loss.shape[0]

    window_starts = list(range(0, max(1, seq_len - occluded_window_size + 1), stride))
    sensitivity = np.zeros((n_angles, len(window_starts)), dtype=np.float32)

    for idx, start in enumerate(tqdm(window_starts, desc="Window occlusion")):
        end = min(seq_len, start + occluded_window_size)
        ablated_loss_sum = None
        n_batches = 0
        with torch.no_grad():
            for Xb, Yb, springback, experiment_config in val_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                springback = springback.to(device)
                experiment_config = experiment_config.to(device)

                Xb_ablated = Xb.clone()
                Xb_ablated[:, start:end, :] = 0

                pred, _ = model(Xb_ablated, springback, experiment_config)
                mse_per_angle = (pred - Yb).pow(2).mean(dim=(0, 2))
                ablated_loss_sum = (
                    mse_per_angle if ablated_loss_sum is None else ablated_loss_sum + mse_per_angle
                )
                n_batches += 1

        ablated_loss = (ablated_loss_sum / n_batches).detach().cpu().numpy()
        sensitivity[:, idx] = ablated_loss - baseline_loss

    df = pd.DataFrame(
        sensitivity,
        index=[f"angle_{i}" for i in range(n_angles)],
        columns=[f"w{start}" for start in window_starts],
    )
    csv_path = saving_dir / "timestep_window_occlusion_sensitivity.csv"
    df.to_csv(csv_path)

    heatmap_path = _save_window_heatmap(
        sensitivity,
        saving_dir,
        "timestep_window_occlusion_sensitivity.png",
        f"Window Sensitivity per Angle (size={occluded_window_size}, stride={stride})",
        window_starts,
        occluded_window_size,
        annot_timesteps,
        mandrel_extraction_annot_timesteps,
        sensor_data,
        sensor_names,
    )
    return df, heatmap_path


def compute_timestep_permutation_sensitivity(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    saving_dir: Path,
    annot_timesteps: Optional[list[int]] = None,
    mandrel_extraction_annot_timesteps: Optional[list[int]] = None,
    sensor_data: Optional[np.ndarray] = None,
    sensor_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Compute per-timestep sensitivity by permuting timestep values across the batch.
    """
    model.eval()
    saving_dir.mkdir(parents=True, exist_ok=True)

    baseline_loss_sum: Optional[torch.Tensor] = None
    n_batches = 0
    seq_len = None

    with torch.no_grad():
        for Xb, Yb, springback, experiment_config in val_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            springback = springback.to(device)
            experiment_config = experiment_config.to(device)

            pred, _ = model(Xb, springback, experiment_config)
            mse_per_angle = (pred - Yb).pow(2).mean(dim=(0, 2))
            baseline_loss_sum = (
                mse_per_angle if baseline_loss_sum is None else baseline_loss_sum + mse_per_angle
            )
            n_batches += 1
            if seq_len is None:
                seq_len = Xb.shape[1]

    if n_batches == 0 or seq_len is None:
        raise ValueError("Validation loader is empty; cannot compute permutation sensitivity.")

    baseline_loss = (baseline_loss_sum / n_batches).detach().cpu().numpy()
    n_angles = baseline_loss.shape[0]
    sensitivity = np.zeros((n_angles, seq_len), dtype=np.float32)

    for t in tqdm(range(seq_len), desc="Timestep permutation"):
        permuted_loss_sum = None
        n_batches = 0
        with torch.no_grad():
            for Xb, Yb, springback, experiment_config in val_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                springback = springback.to(device)
                experiment_config = experiment_config.to(device)

                Xb_perm = Xb.clone()
                perm_indices = torch.randperm(Xb.size(0))
                Xb_perm[:, t, :] = Xb[perm_indices, t, :]

                pred, _ = model(Xb_perm, springback, experiment_config)
                mse_per_angle = (pred - Yb).pow(2).mean(dim=(0, 2))
                permuted_loss_sum = (
                    mse_per_angle if permuted_loss_sum is None else permuted_loss_sum + mse_per_angle
                )
                n_batches += 1

        permuted_loss = (permuted_loss_sum / n_batches).detach().cpu().numpy()
        sensitivity[:, t] = permuted_loss - baseline_loss

    df = pd.DataFrame(
        sensitivity,
        index=[f"angle_{i}" for i in range(n_angles)],
        columns=[f"t{t}" for t in range(seq_len)],
    )
    csv_path = saving_dir / "timestep_permutation_sensitivity.csv"
    df.to_csv(csv_path)

    heatmap_path = _save_timestep_heatmap(
        sensitivity,
        saving_dir,
        "timestep_permutation_sensitivity.png",
        "Timestep Sensitivity per Angle (Permutation)",
        annot_timesteps,
        mandrel_extraction_annot_timesteps,
        sensor_data,
        sensor_names,
    )
    return df, heatmap_path


def compute_timestep_conditional_permutation_sensitivity(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    saving_dir: Path,
    window_size: int,
    annot_timesteps: Optional[list[int]] = None,
    mandrel_extraction_annot_timesteps: Optional[list[int]] = None,
    sensor_data: Optional[np.ndarray] = None,
    sensor_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Permute timesteps within a local window to preserve local statistics.
    """
    model.eval()
    saving_dir.mkdir(parents=True, exist_ok=True)

    baseline_loss_sum: Optional[torch.Tensor] = None
    n_batches = 0
    seq_len = None

    with torch.no_grad():
        for Xb, Yb, springback, experiment_config in val_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            springback = springback.to(device)
            experiment_config = experiment_config.to(device)

            pred, _ = model(Xb, springback, experiment_config)
            mse_per_angle = (pred - Yb).pow(2).mean(dim=(0, 2))
            baseline_loss_sum = (
                mse_per_angle if baseline_loss_sum is None else baseline_loss_sum + mse_per_angle
            )
            n_batches += 1
            if seq_len is None:
                seq_len = Xb.shape[1]

    if n_batches == 0 or seq_len is None:
        raise ValueError("Validation loader is empty; cannot compute conditional permutation.")

    baseline_loss = (baseline_loss_sum / n_batches).detach().cpu().numpy()
    n_angles = baseline_loss.shape[0]
    sensitivity = np.zeros((n_angles, seq_len), dtype=np.float32)
    half = max(1, window_size // 2)

    for t in tqdm(range(seq_len), desc="Conditional permutation"):
        permuted_loss_sum = None
        n_batches = 0
        with torch.no_grad():
            for Xb, Yb, springback, experiment_config in val_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                springback = springback.to(device)
                experiment_config = experiment_config.to(device)

                Xb_perm = Xb.clone()
                start = max(0, t - half)
                end = min(seq_len, t + half + 1)
                window_indices = torch.randint(start, end, (Xb.size(0),), device=Xb.device)
                Xb_perm[:, t, :] = Xb[torch.arange(Xb.size(0), device=Xb.device), window_indices, :]

                pred, _ = model(Xb_perm, springback, experiment_config)
                mse_per_angle = (pred - Yb).pow(2).mean(dim=(0, 2))
                permuted_loss_sum = (
                    mse_per_angle if permuted_loss_sum is None else permuted_loss_sum + mse_per_angle
                )
                n_batches += 1

        permuted_loss = (permuted_loss_sum / n_batches).detach().cpu().numpy()
        sensitivity[:, t] = permuted_loss - baseline_loss

    df = pd.DataFrame(
        sensitivity,
        index=[f"angle_{i}" for i in range(n_angles)],
        columns=[f"t{t}" for t in range(seq_len)],
    )
    csv_path = saving_dir / "timestep_conditional_permutation_sensitivity.csv"
    df.to_csv(csv_path)

    heatmap_path = _save_timestep_heatmap(
        sensitivity,
        saving_dir,
        "timestep_conditional_permutation_sensitivity.png",
        f"Timestep Sensitivity per Angle (Conditional Permutation, w={window_size})",
        annot_timesteps,
        mandrel_extraction_annot_timesteps,
        sensor_data,
        sensor_names,
    )
    return df, heatmap_path


def compute_timestep_masking_sensitivity(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    saving_dir: Path,
    annot_timesteps: Optional[list[int]] = None,
    mandrel_extraction_annot_timesteps: Optional[list[int]] = None,
    sensor_data: Optional[np.ndarray] = None,
    sensor_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Compute per-timestep sensitivity by replacing each timestep with sample mean.
    """
    model.eval()
    saving_dir.mkdir(parents=True, exist_ok=True)

    baseline_loss_sum: Optional[torch.Tensor] = None
    n_batches = 0
    seq_len = None

    with torch.no_grad():
        for Xb, Yb, springback, experiment_config in val_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            springback = springback.to(device)
            experiment_config = experiment_config.to(device)

            pred, _ = model(Xb, springback, experiment_config)
            mse_per_angle = (pred - Yb).pow(2).mean(dim=(0, 2))
            baseline_loss_sum = (
                mse_per_angle if baseline_loss_sum is None else baseline_loss_sum + mse_per_angle
            )
            n_batches += 1
            if seq_len is None:
                seq_len = Xb.shape[1]

    if n_batches == 0 or seq_len is None:
        raise ValueError("Validation loader is empty; cannot compute masking sensitivity.")

    baseline_loss = (baseline_loss_sum / n_batches).detach().cpu().numpy()
    n_angles = baseline_loss.shape[0]
    sensitivity = np.zeros((n_angles, seq_len), dtype=np.float32)

    for t in tqdm(range(seq_len), desc="Timestep masking"):
        masked_loss_sum = None
        n_batches = 0
        with torch.no_grad():
            for Xb, Yb, springback, experiment_config in val_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                springback = springback.to(device)
                experiment_config = experiment_config.to(device)

                Xb_masked = Xb.clone()
                baseline = Xb.mean(dim=1, keepdim=True)
                Xb_masked[:, t, :] = baseline[:, 0, :]

                pred, _ = model(Xb_masked, springback, experiment_config)
                mse_per_angle = (pred - Yb).pow(2).mean(dim=(0, 2))
                masked_loss_sum = (
                    mse_per_angle if masked_loss_sum is None else masked_loss_sum + mse_per_angle
                )
                n_batches += 1

        masked_loss = (masked_loss_sum / n_batches).detach().cpu().numpy()
        sensitivity[:, t] = masked_loss - baseline_loss

    df = pd.DataFrame(
        sensitivity,
        index=[f"angle_{i}" for i in range(n_angles)],
        columns=[f"t{t}" for t in range(seq_len)],
    )
    csv_path = saving_dir / "timestep_masking_sensitivity.csv"
    df.to_csv(csv_path)

    heatmap_path = _save_timestep_heatmap(
        sensitivity,
        saving_dir,
        "timestep_masking_sensitivity.png",
        "Timestep Sensitivity per Angle (Mean Masking)",
        annot_timesteps,
        mandrel_extraction_annot_timesteps,
        sensor_data,
        sensor_names,
    )
    return df, heatmap_path


def run_all_timestep_sensitivity(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    X_sample: torch.Tensor,
    springback_sample: Optional[torch.Tensor],
    experiment_config: Optional[torch.Tensor],
    device: torch.device,
    saving_dir: Path,
    occluded_window_size: int,
    stride: int,
    annot_timesteps: Optional[list[int]] = None,
    mandrel_extraction_annot_timesteps: Optional[list[int]] = None,
    sensor_data: Optional[np.ndarray] = None,
    sensor_names: Optional[list[str]] = None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    ablation_df, ablation_path = compute_timestep_ablation_sensitivity(
        model=model,
        val_loader=val_loader,
        device=device,
        saving_dir=saving_dir,
        annot_timesteps=annot_timesteps,
        mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
        sensor_data=sensor_data,
        sensor_names=sensor_names,
    )
    paths["ablation"] = ablation_path

    ig_df, ig_path = compute_timestep_ig_sensitivity(
        model=model,
        X_sample=X_sample,
        springback_sample=springback_sample,
        experiment_config=experiment_config,
        saving_dir=saving_dir,
        annot_timesteps=annot_timesteps,
        mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
        sensor_data=sensor_data,
        sensor_names=sensor_names,
    )
    paths["ig"] = ig_path

    _, window_path = compute_timestep_window_occlusion_sensitivity(
        model=model,
        val_loader=val_loader,
        device=device,
        saving_dir=saving_dir,
        occluded_window_size=occluded_window_size,
        stride=stride,
        annot_timesteps=annot_timesteps,
        mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
        sensor_data=sensor_data,
        sensor_names=sensor_names,
    )
    paths["window_occlusion"] = window_path

    _, perm_path = compute_timestep_permutation_sensitivity(
        model=model,
        val_loader=val_loader,
        device=device,
        saving_dir=saving_dir,
        annot_timesteps=annot_timesteps,
        mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
        sensor_data=sensor_data,
        sensor_names=sensor_names,
    )
    paths["permutation"] = perm_path

    _, cond_perm_path = compute_timestep_conditional_permutation_sensitivity(
        model=model,
        val_loader=val_loader,
        device=device,
        saving_dir=saving_dir,
        window_size=occluded_window_size,
        annot_timesteps=annot_timesteps,
        mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
        sensor_data=sensor_data,
        sensor_names=sensor_names,
    )
    paths["conditional_permutation"] = cond_perm_path

    _, mask_path = compute_timestep_masking_sensitivity(
        model=model,
        val_loader=val_loader,
        device=device,
        saving_dir=saving_dir,
        annot_timesteps=annot_timesteps,
        mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
        sensor_data=sensor_data,
        sensor_names=sensor_names,
    )
    paths["masking"] = mask_path

    return paths
