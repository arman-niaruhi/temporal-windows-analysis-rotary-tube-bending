import os 
import time
import random
import shutil
import warnings
from tqdm import tqdm
from pathlib import Path
import gc

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from captum.attr import IntegratedGradients

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from src.pipeline.ml.context_extractor.utils.models.attention_lstm import AttentionLSTM
from src.pipeline.ml.context_extractor.utils.data.data_preprocessor import ProcessDataset
from src.pipeline.ml.context_extractor.utils.metrics_helper import (
    compute_all_metrics,
    compute_epoch_metrics,
)
from src.pipeline.ml.context_extractor.utils.visualization_utils import OrganizedImageSaver
from src.pipeline.ml.context_extractor.utils.feature_importance_utils import analyze_feature_importance


def _setup_mlflow_experiment(machine_part: str, params: dict, 
                           preprocessing_info: dict, X: torch.Tensor, 
                           Y: torch.Tensor, target_feature_names: list[str]) -> tuple:
    """Setup MLflow experiment and return experiment description."""
    if mlflow.active_run() is not None:
        mlflow.end_run()
    
    mlflow.set_experiment("LSTM_Attention-All")
    mlflow.set_tracking_uri("mlruns")
    
    N_EXPERIMENTS, TIMESTEPS_IN, FEATURES_IN = X.shape
    N_EXPERIMENTS, N_CROSSCUT, FEATURES_OUT = Y.shape
    
    experiment_description = f"""
    {machine_part} PART - LSTM Attention Model
    ============== PREPROCESSING INFO ====================
    {preprocessing_info}
    ==================== MODEL INFO ======================
    INPUT OF TRAIN:
    N_EXPERIMENTS, N_TIMESTEPS_IN, N_FEATURES_IN = ({N_EXPERIMENTS}, {TIMESTEPS_IN}, {FEATURES_IN})
    OUTPUT_VALIDATION:
    N_EXPERIMENTS, N_CROSSCUT, FEATURES_OUT = ({N_EXPERIMENTS}, {N_CROSSCUT}, {FEATURES_OUT})
   
    GEOMETRY_FEATURES: {target_feature_names}
    ================== TRAINING INFO =====================
    {params}
    """
    return experiment_description, FEATURES_IN, N_CROSSCUT, FEATURES_OUT


def _save_experiment_description_as_text(description: str) -> None:
    """Save experiment description as a text file in MLflow artifacts."""
    desc_path = Path("experiment_description.txt")
    with open(desc_path, "w") as f:
        f.write(description)
    mlflow.log_artifact(str(desc_path))
    desc_path.unlink()
    logger.info("Experiment description saved to MLflow artifacts.")
    

def _create_data_loaders(X_train: torch.Tensor, Y_train: torch.Tensor,
                        X_val: torch.Tensor, Y_val: torch.Tensor,
                        batch_size: int) -> tuple:
    """Create PyTorch DataLoaders for training and validation."""
    train_ds = ProcessDataset(X_train, Y_train)
    val_ds = ProcessDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    plot_loader = DataLoader(val_ds, batch_size=min(64, len(val_ds)), shuffle=False)
    
    return train_loader, val_loader, plot_loader


def _get_plot_batch(plot_loader: DataLoader, device: torch.device) -> tuple:
    """Get a batch of data for plotting."""
    plot_X, plot_Y = next(iter(plot_loader))
    plot_X = plot_X.to(device)
    return plot_X, plot_Y


def _compute_plot_limits(Y_val: torch.Tensor) -> tuple:
    """Compute global y-limits for consistent plotting."""
    y_all = Y_val[:, :, 0].cpu().numpy()
    global_ymin, global_ymax = y_all.min(), y_all.max()
    margin = (global_ymax - global_ymin) * 0.1
    return (global_ymin - margin, global_ymax + margin)


def _log_model_parameters(model: nn.Module) -> None:
    """Log model parameter counts to MLflow."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlflow.log_param("total_parameters", total_params)
    mlflow.log_param("trainable_parameters", trainable_params)


def _log_epoch_metrics(epoch: int, train_loss: float, val_loss: float, 
                     metrics: dict, current_lr: float, epoch_time: float) -> None:
    """Log all metrics for a single epoch to MLflow."""
    mlflow.log_metrics(
        {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mse": metrics["mse"],
            "val_rmse": metrics["rmse"],
            "val_mae": metrics["mae"],
            "val_r2": metrics["r2"],
            "val_mape": metrics["mape"],
            "val_max_error": metrics["max_error"],
            "val_evs": metrics["evs"],
            "val_mbe": metrics["mbe"],
            "val_medae": metrics["medae"],
            "learning_rate": current_lr,
            "epoch_time": epoch_time,
        },
        step=epoch,
    )


def _log_final_metrics(all_targets: torch.Tensor, all_preds: torch.Tensor, 
                     val_losses: list, epoch_times: list) -> None:
    """Compute and log final evaluation metrics to MLflow."""
    final_metrics = compute_all_metrics(all_targets, all_preds)
    
    metrics_to_log = {
        "final_mse": final_metrics["mse"],
        "final_rmse": final_metrics["rmse"],
        "final_mae": final_metrics["mae"],
        "final_r2": final_metrics["r2"],
        "final_max_error": final_metrics["max_error"],
        "final_mean_error": final_metrics["mean_error"],
        "final_std_error": final_metrics["std_error"],
        "total_epochs": len(val_losses),
        "avg_epoch_time": np.mean(epoch_times),
    }

    if "per_feature_mse" in final_metrics:
        for i, (mse, mae) in enumerate(
            zip(final_metrics["per_feature_mse"], final_metrics["per_feature_mae"])
        ):
            metrics_to_log[f"final_mse_feature_{i}"] = mse
            metrics_to_log[f"final_mae_feature_{i}"] = mae

    mlflow.log_metrics(metrics_to_log)


def _create_model(input_features: int, n_predictions: int, output_features: int,
                hidden_dim: int, lstm_layers: int, dropout: float,
                device: torch.device) -> AttentionLSTM:
    """Create and initialize the Attention LSTM model.""" 
    return AttentionLSTM(
            input_features=input_features,
            n_predictions=n_predictions,
            output_features=output_features,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
        ).to(device)
    

def _train_one_epoch(model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module,
                   device: torch.device) -> float:
    """Execute one training epoch."""
    model.train()
    train_loss = 0.0
    
    for Xb, Yb in train_loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        pred, _ = model(Xb)
        loss = criterion(pred, Yb)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        del pred, loss, Xb, Yb
        
    gc.collect()
    torch.cuda.empty_cache()
  
    return train_loss / len(train_loader)


def _validate_one_epoch(model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module, device: torch.device) -> tuple:
    """Execute one validation epoch and collect predictions."""
    model.eval()
    val_loss = 0.0
    val_preds_epoch = []
    val_targets_epoch = []

    with torch.no_grad():
        for Xb, Yb in val_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            pred, _ = model(Xb)
            val_loss += criterion(pred, Yb).item()
            
            val_preds_epoch.append(pred.cpu())
            val_targets_epoch.append(Yb.cpu())

    val_loss /= len(val_loader)
    val_preds_epoch = torch.cat(val_preds_epoch, dim=0)
    val_targets_epoch = torch.cat(val_targets_epoch, dim=0)
    
    return val_loss, val_preds_epoch, val_targets_epoch


def _update_best_model(val_loss: float, best_val_loss: float, model: nn.Module,
                     patience: int, epoch: int) -> tuple:
    """Update best model state if validation loss improved."""
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_state = model.state_dict()
        patience = 0
        mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
    else:
        patience += 1
        best_state = None
    
    return best_val_loss, best_state, patience


def _should_save_plots(epoch: int) -> bool:
    """Determine if plots should be saved this epoch."""
    return epoch % 2 == 0 or epoch == 1


def _generate_epoch_plots(model: nn.Module, plot_X: torch.Tensor, plot_Y: torch.Tensor,
                        X_train: torch.Tensor, sensor_names: list, 
                        target_feature_names: list, val_losses: list, 
                        train_losses: list, epoch: int, image_saver: OrganizedImageSaver,
                        y_lim: tuple, predictions_out: int, train_loss: float,
                        val_loss: float, best_val_loss: float, 
                        annot_timesteps: list, mandrel_extraction_annot_timesteps: list,
                        n_samples: int = 4) -> None:
    """Generate and save plots for the current epoch."""
    with torch.no_grad():
        pred, attn = model(plot_X)
        pred_np = pred.cpu().numpy()
        true_np = plot_Y.cpu().numpy()
        attn_mean = attn.mean(0).cpu().numpy()

    idxs = random.sample(range(len(true_np)), min(n_samples, len(true_np)))
    x_axis = np.arange(predictions_out)
    
    pred_data = (true_np, pred_np, idxs)
    loss_data = (list(range(1, len(val_losses) + 1)), val_losses, train_losses)
    attn_data = attn_mean

    image_saver.save_epoch_plots(
        X_train, sensor_names, target_feature_names,
        pred_data, loss_data, attn_data,
        epoch, x_axis, y_lim, predictions_out,
        train_loss, val_loss, best_val_loss,
        annot_timesteps, mandrel_extraction_annot_timesteps
    )


def _format_progress_bar(train_loss: float, val_loss: float, metrics: dict,
                       best_val_loss: float, current_lr: float, patience: int) -> dict:
    """Format metrics for progress bar display."""
    return {
        "Train": f"{train_loss:.6f}",
        "Val": f"{val_loss:.6f}",
        "MSE": f"{metrics['mse']:.6f}",
        "MAE": f"{metrics['mae']:.6f}",
        "R²": f"{metrics['r2']:.4f}",
        "MAPE": f"{metrics['mape']:.2f}%",
        "MedAE": f"{metrics['medae']:.6f}",
        "Best": f"{best_val_loss:.6f}",
        "LR": f"{current_lr:.2e}",
        "Patience": f"{patience}/10",
    }


def _evaluate_final_model(model: nn.Module, val_loader: DataLoader, 
                        device: torch.device) -> tuple:
    """Evaluate model on full validation set."""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for Xb, Yb in val_loader:
            Xb = Xb.to(device)
            pred, _ = model(Xb)
            all_preds.append(pred.cpu())
            all_targets.append(Yb)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return all_targets, all_preds


def _log_feature_importance_to_mlflow(combined_importance_df: dict, 
                                    importance_paths: dict) -> None:
    """Log feature importance results to MLflow."""
    if combined_importance_df is None:
        return
    
    combined_csv_path = importance_paths.get("combined_csv")
    if combined_csv_path and Path(combined_csv_path).exists():
        mlflow.log_artifact(str(combined_csv_path))
    else:
        logger.warning("Combined CSV path missing or does not exist; skipping MLflow log.")
    
    combined_importance_df = {
        k: v if isinstance(v, list) else [v]
        for k, v in combined_importance_df.items()
    }
    df = pd.DataFrame(combined_importance_df)
    df.to_csv("feature_importance_summary.csv", index=False)
    mlflow.log_artifact("feature_importance_summary.csv")
    Path("feature_importance_summary.csv").unlink()


def _generate_final_attention_plot(model: nn.Module, plot_X: torch.Tensor,
                                 X_val: torch.Tensor, sensor_names: list,
                                 image_saver: OrganizedImageSaver,
                                 annot_timesteps: list,
                                 mandrel_extraction_annot_timesteps: list) -> None:
    """Generate final attention visualization."""
    with torch.no_grad():
        model.eval()
        _, final_attn = model(plot_X)
        final_attn_mean = final_attn.mean(0).cpu().numpy()

    image_saver.plot_attention_lines_with_sensors(
        sensor_data=X_val,
        sensor_names=sensor_names,
        attn_mean=final_attn_mean,
        annot_timesteps=annot_timesteps,
        mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
        sample_idx=-1,
    )


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
                image_saver: OrganizedImageSaver) -> None:
    """Save Integrated Gradients attributions to CSV files."""
    cleaned_sensor_names = [name.replace("_mean", "") for name in sensor_names]
    
    for idx, attributions in enumerate(ig_maps):
        attr_df = pd.DataFrame(attributions, columns=cleaned_sensor_names)
        target_name = target_feature_names[idx] if target_feature_names else idx
        csv_path = image_saver.base_dir / f"ig_feature_{target_name}.csv"
        attr_df.to_csv(csv_path, index=False)


def __create_sensor_plot_axis(ax: plt.Axes, sample_data: np.ndarray, 
                           sensor_names: list, colors: np.ndarray,
                           machine_part: str,
                           annot_timesteps: list = None,
                           mandrel_extraction_annot_timesteps: list = None) -> None:
    """Create sensor data plot on given axis."""
    cleaned_names = [name.replace("_mean", "") for name in sensor_names]
    main_timesteps = sample_data.shape[0]
    
    for i, color in enumerate(colors):
        ax.plot(sample_data[:, i], color=color, linewidth=2.5, alpha=0.85,
                label=cleaned_names[i], marker="o", markersize=3,
                markevery=max(1, main_timesteps // 20))
    
    ax.set_xlabel("Time Step", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sensor Value", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.set_facecolor("#f9f9f9")
    ax.set_xlim(0, main_timesteps-1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    if annot_timesteps and machine_part == "All":
        annot_labels = ["Start-Declamping", "Start-Bending", 
                       "Start-Declamping", "End-Declamping"]
        for ts, label in zip(annot_timesteps, annot_labels):
            ax.axvline(ts, color="black", linestyle="--", alpha=0.7)
            ax.annotate(label, xy=(ts, sample_data.max()), xytext=(0,10),
                       textcoords="offset points", ha="center", va="bottom",
                       fontsize=11, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8))
    
    if mandrel_extraction_annot_timesteps and machine_part == "All":
        ax.axvspan(mandrel_extraction_annot_timesteps[0],
                  mandrel_extraction_annot_timesteps[1],
                  color="blue", alpha=0.12, linewidth=0)


def __create_ig_heatmap_axis(ax: plt.Axes, attributions: np.ndarray,
                          sensor_names: list, target_name: str,
                          main_timesteps: int) -> None:
    """Create Integrated Gradients heatmap on given axis."""
    cleaned_names = [name.replace("_mean", "") for name in sensor_names]
    cleaned_names_reversed = list(reversed(cleaned_names))
    
    im = ax.imshow(attributions.T, cmap="magma", aspect="auto",
                  interpolation="nearest",
                  extent=[0, main_timesteps-1, 0, len(cleaned_names_reversed)])
    
    ax.set_yticks(np.arange(len(cleaned_names_reversed)) + 0.5)
    ax.set_yticklabels(cleaned_names_reversed)
    ax.set_xlim(0, main_timesteps-1)
    ax.set_xlabel("Time Step", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{target_name}", fontsize=12, fontweight="bold")
    ax.set_facecolor("white")
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(1.2)


def __save_combined_ig_plot(ig_maps: list, sample_data: np.ndarray, 
                         sensor_names: list, target_feature_names: list,
                         image_saver: OrganizedImageSaver, colors: np.ndarray, machine_part,
                         annot_timesteps: list = None,
                         mandrel_extraction_annot_timesteps: list = None,
                         figsize: tuple = (25, 3)) -> None:
    """Save combined Integrated Gradients plot with all features."""
    n_output_features = len(ig_maps)
    main_timesteps = sample_data.shape[0]
    
    n_rows = n_output_features + 1
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_rows), facecolor="white")
    gs = fig.add_gridspec(n_rows, 1, height_ratios=[2] + [1]*n_output_features, hspace=0.25)
    
    ax_main = fig.add_subplot(gs[0])
    __create_sensor_plot_axis(ax_main, sample_data, sensor_names, colors, machine_part,
                           annot_timesteps, mandrel_extraction_annot_timesteps)
    ax_main.legend(loc="upper left", bbox_to_anchor=(1.02, 1), 
                  frameon=True, fontsize=10)
    
    for idx, attributions in enumerate(ig_maps):
        ax = fig.add_subplot(gs[idx+1])
        target_name = target_feature_names[idx] if target_feature_names else f"Feature {idx}"
        __create_ig_heatmap_axis(ax, attributions, sensor_names, target_name, main_timesteps)
    
    plt.tight_layout()
    combined_path = image_saver.base_dir / "ig_combined.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(combined_path))


def __save_individual_ig_plots(ig_maps: list, sample_data: np.ndarray,
                            sensor_names: list, target_feature_names: list,
                            image_saver: OrganizedImageSaver, colors: np.ndarray, machine_part,
                            annot_timesteps: list = None,
                            mandrel_extraction_annot_timesteps: list = None) -> None:
    """Save individual Integrated Gradients plots for each feature."""
    main_timesteps = sample_data.shape[0]
    
    for idx, attributions in enumerate(ig_maps):
        target_name = target_feature_names[idx] if target_feature_names else f"Feature_{idx}"
        feature_folder = image_saver.base_dir / target_name.replace(" ", "_")
        os.makedirs(feature_folder, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 10), facecolor="white")
        gs = fig.add_gridspec(2, 2, width_ratios=[0.88, 0.12],
                            height_ratios=[2, 1], hspace=0.3, wspace=0.05)
        
        ax_top = fig.add_subplot(gs[0, 0])
        __create_sensor_plot_axis(ax_top, sample_data, sensor_names, colors, machine_part,
                               annot_timesteps, mandrel_extraction_annot_timesteps)
        
        legend_ax = fig.add_subplot(gs[0, 1])
        legend_ax.axis('off')
        handles, labels = ax_top.get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc='upper left', fontsize=9,
                        frameon=True, borderpad=0.8, labelspacing=0.5,
                        handlelength=1.5, handletextpad=0.5, borderaxespad=0.5)
        
        ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
        __create_ig_heatmap_axis(ax_bottom, attributions, sensor_names, 
                              target_name, main_timesteps)
        
        cbar_ax = fig.add_subplot(gs[1, 1])
        cleaned_names_reversed = list(reversed([n.replace("_mean", "") for n in sensor_names]))
        im = ax_bottom.imshow(attributions.T, aspect="auto", cmap="magma",
                             interpolation="nearest",
                             extent=[0, main_timesteps-1, 0, len(cleaned_names_reversed)])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_linewidth(1.2)
        
        plt.tight_layout()
        indiv_path = feature_folder / "ig.png"
        fig.savefig(indiv_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(indiv_path))


def _save_integrated_gradients_combined(
    model: torch.nn.Module, X_sample: torch.Tensor,
    sensor_data: torch.Tensor, sensor_names: list[str],
    target_feature_names: list[str],
    image_saver: OrganizedImageSaver,
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
    
    __save_ig_csvs(ig_maps, sensor_names, target_feature_names, image_saver)
    
    __save_combined_ig_plot(ig_maps, sample_data, sensor_names, target_feature_names,
                         image_saver, colors, machine_part, annot_timesteps,
                         mandrel_extraction_annot_timesteps, figsize_combined)
    
    __save_individual_ig_plots(ig_maps, sample_data, sensor_names, target_feature_names,
                            image_saver, colors, machine_part, annot_timesteps,
                            mandrel_extraction_annot_timesteps)


def __create_metric_plot(epochs: list, values: list, ylabel: str, title: str,
                      color: str, output_path: Path, reference_line: dict = None) -> None:
    """Create and save a single metric plot."""
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, values, color=color, linewidth=2.5, marker='o', markersize=4)
    
    if reference_line:
        ax.axhline(y=reference_line['y'], color=reference_line.get('color', 'gray'),
                  linestyle='--', alpha=0.5, linewidth=2,
                  label=reference_line.get('label', ''))
        ax.legend(fontsize=12)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close("all")
    gc.collect()
    logger.info(f"Saved {title} plot to {output_path}")


def __plot_loss_curves(epochs: list, train_losses: list, val_losses: list,
                    image_saver: OrganizedImageSaver) -> Path:
    """Create and save training and validation loss curves."""
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label='Train Loss', color='blue',
           linewidth=2.5, marker='o', markersize=4)
    ax.plot(epochs, val_losses, label='Val Loss', color='red',
           linewidth=2.5, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training and Validation Loss', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = image_saver.base_dir / "metric_loss.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close("all")
    gc.collect()
    logger.info(f"Saved Loss plot to {path}")
    return path


def __create_metrics_summary_file(metrics_history: dict, train_losses: list,
                               val_losses: list, epoch_times: list,
                               learning_rates: list, image_saver: OrganizedImageSaver) -> Path:
    """Create a text file summarizing all training metrics."""
    epochs = list(range(1, len(train_losses) + 1))
    summary_path = image_saver.base_dir / "metrics_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FINAL TRAINING METRICS SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Epochs:          {len(epochs)}\n")
        f.write(f"Best Val Loss:         {min(val_losses):.6f}\n")
        f.write(f"Final Val Loss:        {val_losses[-1]:.6f}\n")
        f.write(f"Final Train Loss:      {train_losses[-1]:.6f}\n\n")
        f.write("-"*60 + "\n")
        f.write("FINAL VALIDATION METRICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"MSE:                   {metrics_history['mse'][-1]:.6f}\n")
        f.write(f"RMSE:                  {metrics_history['rmse'][-1]:.6f}\n")
        f.write(f"MAE:                   {metrics_history['mae'][-1]:.6f}\n")
        f.write(f"MedAE:                 {metrics_history['medae'][-1]:.6f}\n")
        f.write(f"R² Score:              {metrics_history['r2'][-1]:.6f}\n")
        f.write(f"MAPE:                  {metrics_history['mape'][-1]:.2f}%\n")
        f.write(f"Max Error:             {metrics_history['max_error'][-1]:.6f}\n")
        f.write(f"EVS:                   {metrics_history['evs'][-1]:.6f}\n")
        f.write(f"MBE:                   {metrics_history['mbe'][-1]:.6f}\n\n")
        f.write("-"*60 + "\n")
        f.write("TRAINING STATISTICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"Avg Epoch Time:        {np.mean(epoch_times):.2f}s\n")
        f.write(f"Total Training Time:   {sum(epoch_times):.2f}s\n")
        f.write(f"Final Learning Rate:   {learning_rates[-1]:.2e}\n")
        f.write("="*60 + "\n")
    
    logger.info(f"Saved Metrics Summary to {summary_path}")
    return summary_path


def _plot_all_metrics(metrics_history: dict, train_losses: list[float],
                    val_losses: list[float], learning_rates: list[float],
                    epoch_times: list[float], image_saver: OrganizedImageSaver) -> None:
    """Create individual plots for each training metric."""
    epochs = list(range(1, len(train_losses) + 1))
    saved_paths = []
    
    path = __plot_loss_curves(epochs, train_losses, val_losses, image_saver)
    saved_paths.append(path)
    
    path = image_saver.base_dir / "metric_mse.png"
    __create_metric_plot(epochs, metrics_history['mse'], 'MSE',
                      'Mean Squared Error', 'purple', path)
    saved_paths.append(path)
    
    path = image_saver.base_dir / "metric_rmse.png"
    __create_metric_plot(epochs, metrics_history['rmse'], 'RMSE',
                      'Root Mean Squared Error', 'darkviolet', path)
    saved_paths.append(path)
    
    path = image_saver.base_dir / "metric_mae.png"
    __create_metric_plot(epochs, metrics_history['mae'], 'MAE',
                      'Mean Absolute Error', 'orange', path)
    saved_paths.append(path)
    
    path = image_saver.base_dir / "metric_medae.png"
    __create_metric_plot(epochs, metrics_history['medae'], 'MedAE',
                      'Median Absolute Error', 'darkorange', path)
    saved_paths.append(path)
    
    path = image_saver.base_dir / "metric_r2.png"
    __create_metric_plot(epochs, metrics_history['r2'], 'R² Score',
                      'R² Score (Coefficient of Determination)', 'green', path,
                      reference_line={'y': 1.0, 'label': 'Perfect Score'})
    saved_paths.append(path)
    
    path = image_saver.base_dir / "metric_mape.png"
    __create_metric_plot(epochs, metrics_history['mape'], 'MAPE (%)',
                      'Mean Absolute Percentage Error', 'brown', path)
    saved_paths.append(path)
    
    path = image_saver.base_dir / "metric_max_error.png"
    __create_metric_plot(epochs, metrics_history['max_error'], 'Max Error',
                      'Maximum Error', 'red', path)
    saved_paths.append(path)
    
    path = image_saver.base_dir / "metric_evs.png"
    __create_metric_plot(epochs, metrics_history['evs'], 'EVS',
                      'Explained Variance Score', 'teal', path,
                      reference_line={'y': 1.0, 'label': 'Perfect Score'})
    saved_paths.append(path)
    
    path = image_saver.base_dir / "metric_mbe.png"
    __create_metric_plot(epochs, metrics_history['mbe'], 'MBE',
                      'Mean Bias Error', 'navy', path,
                      reference_line={'y': 0, 'label': 'Zero Bias'})
    saved_paths.append(path)
    
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, learning_rates, color='magenta', linewidth=2.5,
           marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_learning_rate.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close("all")
    gc.collect()
    saved_paths.append(path)
    logger.info(f"Saved Learning Rate plot to {path}")
    
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, epoch_times, color='cyan', linewidth=2.5,
           marker='o', markersize=4)
    avg_time = np.mean(epoch_times)
    ax.axhline(y=avg_time, color='red', linestyle='--', alpha=0.5,
              linewidth=2, label=f'Avg: {avg_time:.2f}s')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_title('Training Time per Epoch', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_epoch_time.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close("all")
    gc.collect()
    saved_paths.append(path)
    logger.info(f"Saved Epoch Time plot to {path}")
    
    summary_path = __create_metrics_summary_file(
        metrics_history, train_losses, val_losses, epoch_times,
        learning_rates, image_saver
    )
    saved_paths.append(summary_path)
    
    for path in saved_paths:
        mlflow.log_artifact(str(path))
    
    logger.info(f"Total of {len(saved_paths)} metric files saved and logged to MLflow")


def _move_images_to_mlflow_artifacts(image_saver: OrganizedImageSaver) -> None:
    """Move entire image folder to MLflow experiment artifacts directory."""
    try:
        base_dir = image_saver.base_dir
        run = mlflow.active_run()
        
        if run is None:
            logger.warning("No active MLflow run found. Cannot log images to MLflow artifacts.")
            return None

        if base_dir.exists():
            mlflow.log_artifact(str(base_dir))
            shutil.rmtree(base_dir)
            return True
        else:
            logger.warning(f"Image directory {base_dir} does not exist. Skipping MLflow logging.")
            return None

    except Exception as e:
        logger.error(f"Error logging images to MLflow: {e}")
        return None


def _window_based_importance(model, train_loader,
                             occluded_window_size: int = 10, stride: int = 5, 
                             device=None, n_samples: int = 2):
    """
    Compute window-based importance:
    - Mean importance across all samples
    - Importance for n_samples for visualization
    Returns:
        importance_df: DataFrame of mean importance
        mean_importance: np.ndarray (n_windows,)
        sample_importances: list of np.ndarray (per-sample importance)
        sample_data: list of np.ndarray (original data for selected samples)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_importance = []
    n_windows = None
    sample_importances = []
    sample_data = []

    with torch.no_grad():
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            batch_size, T, F = X_batch.shape

            for b in range(batch_size):
                x = X_batch[b:b+1].clone()
                original_pred, _ = model(x)
                original_pred = original_pred.cpu().numpy()

                importance_vals = []

                for start in range(0, T - occluded_window_size + 1, stride):
                    x_occluded = x.clone()
                    x_occluded[:, start:start+occluded_window_size, :] = 0.0
                    occluded_pred, _ = model(x_occluded)
                    occluded_pred = occluded_pred.cpu().numpy()

                    delta = np.mean(np.abs(original_pred - occluded_pred))
                    importance_vals.append(delta)

                all_importance.append(importance_vals)
                if n_windows is None:
                    n_windows = len(importance_vals)

                # Save the first n_samples for visualization
                if len(sample_importances) < n_samples:
                    sample_importances.append(np.array(importance_vals))
                    sample_data.append(x.squeeze(0).cpu().numpy())

    all_importance = np.array(all_importance)
    mean_importance = all_importance.mean(axis=0)
    importance_df = pd.DataFrame([mean_importance], columns=[f"window_{i}" for i in range(n_windows)])

    return importance_df, mean_importance, sample_importances, sample_data


def _visualize_window_importance(sample_importances: list[np.ndarray],
                                 sample_data: list[np.ndarray],
                                 feature_names: list[str],
                                 image_saver: OrganizedImageSaver,
                                 top_k_windows: int = 5):
    """
    Plot original sample data, sample importance, and mean importance across all samples.
    3 subplots: Top = original data, Middle = sample importance, Bottom = mean importance.
    """
    n_samples = len(sample_importances)
    T = sample_data[0].shape[0]  # assuming all samples same length

    # Compute mean importance across all samples
    n_windows = len(sample_importances[0])
    mean_importance = np.mean(sample_importances, axis=0)
    window_size = T // n_windows
    mean_importance_full = np.repeat(mean_importance, window_size)
    if len(mean_importance_full) < T:
        mean_importance_full = np.pad(mean_importance_full, (0, T - len(mean_importance_full)), mode='edge')

    # Identify top_k windows for mean importance
    top_windows_idx = np.argsort(mean_importance)[-top_k_windows:]
    top_windows_steps = []
    for w in top_windows_idx:
        start = w * window_size
        end = min((w + 1) * window_size, T)
        top_windows_steps.extend(range(start, end))
    top_windows_steps = sorted(top_windows_steps)

    colors = plt.cm.tab20(np.linspace(0, 1, sample_data[0].shape[1]))

    for i in range(n_samples):
        data = sample_data[i]  # (T, F)
        importance = sample_importances[i]
        importance_full = np.repeat(importance, window_size)
        if len(importance_full) < T:
            importance_full = np.pad(importance_full, (0, T - len(importance_full)), mode='edge')

        fig, axes = plt.subplots(3, 1, figsize=(20, 10), gridspec_kw={'height_ratios':[2, 1, 1]})

        # --- Top: original multivariate data ---
        for f_idx in range(data.shape[1]):
            axes[0].plot(data[:, f_idx], label=feature_names[f_idx], color=colors[f_idx], linewidth=1.5)
        axes[0].set_title(f"Sample {i} Original Data")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Value")
        axes[0].grid(True, linestyle='--', alpha=0.2)
        axes[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

        # --- Middle: this sample importance ---
        axes[1].fill_between(range(T), 0, importance_full, color='salmon', alpha=0.3, label='Importance')
        # Highlight top windows for this sample
        top_idx_sample = np.argsort(importance)[-top_k_windows:]
        top_mask_sample = np.zeros(T, dtype=bool)
        for w in top_idx_sample:
            start = w * window_size
            end = min((w + 1) * window_size, T)
            top_mask_sample[start:end] = True
        axes[1].fill_between(range(T), 0, importance_full, where=top_mask_sample,
                             color='red', alpha=0.5, edgecolor='darkred', linewidth=2,
                             label=f'Top {top_k_windows} Windows')
        axes[1].set_title(f"Sample {i} Window-Based Importance")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Importance")
        axes[1].grid(True, linestyle='--', alpha=0.2)
        axes[1].legend(loc='upper right', fontsize=10)

        # --- Bottom: mean importance across all samples ---
        axes[2].fill_between(range(T), 0, mean_importance_full, color='lightblue', alpha=0.3, label='Mean Importance')
        # Highlight top windows for mean
        if len(top_windows_steps) > 0:
            top_mask_mean = np.zeros(T, dtype=bool)
            top_mask_mean[top_windows_steps] = True
            axes[2].fill_between(range(T), 0, mean_importance_full, where=top_mask_mean,
                                 color='blue', alpha=0.5, edgecolor='darkblue', linewidth=2,
                                 label=f'Top {top_k_windows} Mean Windows')
        axes[2].set_title("Mean Window-Based Importance Across Samples")
        axes[2].set_xlabel("Time Step")
        axes[2].set_ylabel("Importance")
        axes[2].grid(True, linestyle='--', alpha=0.2)
        axes[2].legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        path = image_saver.base_dir / f"window_importance_sample_{i}_mean.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close("all")
        gc.collect()
        mlflow.log_artifact(str(path))
        logger.info(f"Window importance plot for sample {i} saved with mean importance.")
        

def _save_window_importance_results(importance_df: pd.DataFrame,
                                   window_importance_map: np.ndarray,
                                   image_saver: OrganizedImageSaver) -> None:
    """
    Save mean window importance results to CSV and MLflow.
    """
    csv_path = image_saver.base_dir / "window_importance_mean.csv"
    importance_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path))
    logger.info(f"Mean window-based importance saved to {csv_path}")
    

def find_previous_mlflow_run(machine_part: str, preprocessing_info: dict):
    """
    Automatically find the latest MLflow model based on run_name tag:
    {machine_part}_{excluded58}_ws{window_size}.
    Returns model URI if found, else None.
    """
    client = MlflowClient()
    experiment_name = "LSTM_Attention-All"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"No MLflow experiment found with name {experiment_name}.")
        return None, None

    # Construct run_name pattern
    excluded58 = "" if not preprocessing_info.get('to_58_excluded', False) else "58"
    window_size = str(preprocessing_info.get('window_num', '0'))
    run_name_to_search = f"{machine_part}_{excluded58}_ws{window_size}"
    # Search runs by run_name tag
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name_to_search}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if runs:
        run = runs[0]
        logger.info(f"Found previous run: {run.info.run_name} (run_id: {run.info.run_id})")
        model_uri = f"runs:/{run.info.run_id}/model"
        return run.info.run_id, model_uri

    logger.info(f"No previous run found with run_name: {run_name_to_search}")
    return None, None

def train_model(
    X: torch.Tensor,
    Y: torch.Tensor,
    params: dict,
    occlusion_params: dict,
    sensor_names: list[str],
    target_feature_names: list[str],
    machine_part: str,
    preprocessing_info: dict,
    annot_timesteps: list[int],
    mandrel_extraction_annot_timesteps: list[int],
) -> None:
    """
    Train Attention LSTM model or reuse an existing trained model.
    If train=False, resumes the original MLflow run and replaces plots.
    """

    logger.info("Starting train_model")
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    excluded58 = "" if not preprocessing_info["to_58_excluded"] else "58"
    window_size = str(preprocessing_info["window_num"])

    # ----------------------------
    # Data + experiment setup
    # ----------------------------
    experiment_desc, features_in, predictions_out, features_out = \
        _setup_mlflow_experiment(
            machine_part, params, preprocessing_info, X, Y, target_feature_names
        )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )

    train_loader, val_loader, plot_loader = _create_data_loaders(
        X_train, Y_train, X_val, Y_val, params["batch_size"]
    )

    plot_X, plot_Y = _get_plot_batch(plot_loader, device)
    n_samples = min(4, len(plot_Y))
    y_lim = _compute_plot_limits(Y_val)

    # ============================================================
    # CASE 1 — TRAIN = FALSE → RESUME EXISTING RUN
    # ============================================================
    if not params.get("train"):
        run_id, model_uri = find_previous_mlflow_run(
            machine_part, preprocessing_info
        )

        if run_id is None:
            logger.warning("No matching MLflow run found to resume.")
            return

        logger.info(f"Resuming MLflow run_id={run_id}")

        with mlflow.start_run(run_id=run_id):
            image_saver = OrganizedImageSaver("images", machine_part)

            model = mlflow.pytorch.load_model(model_uri, map_location=device)

            # ---- plots & diagnostics (overwrite artifacts) ----
            _generate_final_attention_plot(
                model, plot_X, X_val, sensor_names, image_saver,
                annot_timesteps, mandrel_extraction_annot_timesteps
            )

            combined_importance_df, _, importance_paths = analyze_feature_importance(
                model=model,
                val_loader=val_loader,
                feature_names=sensor_names
            )

            if combined_importance_df is not None:
                _log_feature_importance_to_mlflow(
                    combined_importance_df, importance_paths
                )

            importance_df, mean_importance, sample_importances, sample_data = \
                _window_based_importance(
                    model=model,
                    train_loader=val_loader,
                    occluded_window_size=params.get("occlusion_window_size", 10),
                    stride=params.get("occlusion_stride", 5),
                    device=device,
                    n_samples=params.get("number_of_samples", 10)
                )
                
            top_k_windows = params.get("top_k_windows_to_plot", 10)

            _visualize_window_importance(
                sample_importances, sample_data, sensor_names, image_saver, top_k_windows
            )
            _save_window_importance_results(
                importance_df, mean_importance, image_saver
            )

            _move_images_to_mlflow_artifacts(image_saver)

        logger.info("Training skipped; plots replaced in original MLflow run.")
        return

    # ============================================================
    # CASE 2 — TRAIN = TRUE → NEW RUN
    # ============================================================
    with mlflow.start_run(run_name=f"{machine_part}_{excluded58}_ws{window_size}"):
        image_saver = OrganizedImageSaver("images", machine_part=machine_part)
        _save_experiment_description_as_text(experiment_desc)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        
        y_lim = _compute_plot_limits(Y_val)
        plot_X, plot_Y = _get_plot_batch(plot_loader, device)
        n_samples = min(4, len(plot_Y))
        
        model = _create_model(features_in, predictions_out, features_out,
                           params["hidden_dim"], params["lstm_layers"],
                           params["dropout"], device)
        _log_model_parameters(model)
        
        optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        val_losses, train_losses, learning_rates, epoch_times = [], [], [], []
        metrics_history = {
            'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'mape': [],
            'max_error': [], 'evs': [], 'mbe': [], 'medae': []
        }
        best_val_loss = float("inf")
        best_state = None
        patience = 0
        
        fpbar = tqdm(range(1, params["max_epochs"] + 1), desc="Training")
        for epoch in fpbar:
            epoch_start = time.time()
            
            train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_preds, val_targets = _validate_one_epoch(
                model, val_loader, criterion, device
            )
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            metrics = compute_epoch_metrics(val_targets, val_preds)
            for key in metrics_history.keys():
                metrics_history[key].append(metrics[key])
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            learning_rates.append(current_lr)
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            _log_epoch_metrics(epoch, train_loss, val_loss, metrics, current_lr, epoch_time)
            
            best_val_loss, new_state, patience = _update_best_model(
                val_loss, best_val_loss, model, patience, epoch
            )
            if new_state is not None:
                best_state = new_state
            
            if _should_save_plots(epoch):
                _generate_epoch_plots(
                    model, plot_X, plot_Y, X_train, sensor_names,
                    target_feature_names, val_losses, train_losses, epoch,
                    image_saver, y_lim, predictions_out, train_loss, val_loss,
                    best_val_loss, annot_timesteps, mandrel_extraction_annot_timesteps,
                    n_samples
                )
            
            progress_info = _format_progress_bar(
                train_loss, val_loss, metrics, best_val_loss, current_lr, patience
            )
            fpbar.set_postfix(progress_info, refresh=True)
            
            if patience >= 10:
                mlflow.log_param("stopped_at_epoch", epoch)
                break
        
        if best_state is not None:
            model.load_state_dict(best_state)
        
        all_targets, all_preds = _evaluate_final_model(model, val_loader, device)
        _log_final_metrics(all_targets, all_preds, val_losses, epoch_times)
        
        mlflow.pytorch.log_model(model.cpu(), "model")
        logger.info("Model training completed and logged to MLflow.")
        
        _plot_all_metrics(metrics_history, train_losses, val_losses,
                        learning_rates, epoch_times, image_saver)
        
        combined_importance_df, _, importance_paths = analyze_feature_importance(
        model=model,
        val_loader=val_loader,
        feature_names=sensor_names
    )
        
        if combined_importance_df is not None:
            X_sample = plot_X[:1]
            sensor_data_sample = X_val[:1].cpu().numpy()
            _save_integrated_gradients_combined(
                model, X_sample, sensor_data_sample, sensor_names,
                target_feature_names, image_saver, machine_part, annot_timesteps,
                mandrel_extraction_annot_timesteps
            )
            
            _log_feature_importance_to_mlflow(combined_importance_df, importance_paths)
        
        _generate_final_attention_plot(
            model, plot_X, X_val, sensor_names, image_saver,
            annot_timesteps, mandrel_extraction_annot_timesteps
        )
        
        importance_df, mean_importance, sample_importances, sample_data = \
                _window_based_importance(
                    model=model,
                    train_loader=val_loader,
                    occluded_window_size=occlusion_params.get("occlusion_window_size", 100),
                    stride=occlusion_params.get("occlusion_stride", 50),
                    device=device,
                    n_samples=occlusion_params.get("number_of_samples", 5)
                )

        _visualize_window_importance(
            sample_importances, sample_data, sensor_names, image_saver
        )
        _save_window_importance_results(
            importance_df, mean_importance, image_saver
        )

        _move_images_to_mlflow_artifacts(image_saver)