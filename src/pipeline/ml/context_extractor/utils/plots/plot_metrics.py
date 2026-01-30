"""
Quantitative performance and convergence metrics.

This module generates plots of loss curves and evaluation metrics across
training epochs, as well as summary files used to report final results.
The figures produced here support the quantitative evaluation presented
in the paper.

No interpretability or attention visualizations are included in this module.
"""


import numpy as np
import logging
import matplotlib.pyplot as plt
import gc
from pathlib import Path


logger = logging.getLogger(__name__)

def __create_metrics_summary_file(metrics_history: dict, train_losses: list,
                               val_losses: list, epoch_times: list,
                               learning_rates: list, saving_dir,
                               split_config_path: str | None = None):
    """Create a text file summarizing all training metrics."""
    epochs = list(range(1, len(train_losses) + 1))
    summary_path = saving_dir / "metrics_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FINAL TRAINING METRICS SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Epochs:          {len(epochs)}\n")
        f.write(f"Best Val Loss:         {min(val_losses):.6f}\n")
        f.write(f"Final Val Loss:        {val_losses[-1]:.6f}\n")
        f.write(f"Final Train Loss:      {train_losses[-1]:.6f}\n")
        if split_config_path:
            f.write(f"Split Config:          {split_config_path}\n")
        f.write("\n")
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


def __plot_loss_curves(epochs: list, train_losses: list, val_losses: list, saving_dir: Path):
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
    
    path = saving_dir / "metric_loss.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close("all")
    gc.collect()


def plot_all_metrics(metrics_history: dict, train_losses: list[float],
                    val_losses: list[float], learning_rates: list[float],
                    epoch_times: list[float], saving_dir: Path,
                    split_config_path: str | None = None) -> None:
    """Create individual plots for each training metric."""
    epochs = list(range(1, len(train_losses) + 1))
    
    __plot_loss_curves(epochs, train_losses, val_losses, saving_dir)
    
    path = saving_dir / "metric_mse.png"
    __create_metric_plot(epochs, metrics_history['mse'], 'MSE',
                      'Mean Squared Error', 'purple', path)
    
    path = saving_dir / "metric_rmse.png"
    __create_metric_plot(epochs, metrics_history['rmse'], 'RMSE',
                      'Root Mean Squared Error', 'darkviolet', path)
    
    path = saving_dir / "metric_mae.png"
    __create_metric_plot(epochs, metrics_history['mae'], 'MAE',
                      'Mean Absolute Error', 'orange', path)

    path = saving_dir / "metric_medae.png"
    __create_metric_plot(epochs, metrics_history['medae'], 'MedAE',
                      'Median Absolute Error', 'darkorange', path)
 
    path = saving_dir / "metric_r2.png"
    __create_metric_plot(epochs, metrics_history['r2'], 'R² Score',
                      'R² Score (Coefficient of Determination)', 'green', path,
                      reference_line={'y': 1.0, 'label': 'Perfect Score'})
    
    path = saving_dir / "metric_mape.png"
    __create_metric_plot(epochs, metrics_history['mape'], 'MAPE (%)',
                      'Mean Absolute Percentage Error', 'brown', path)
    
    path = saving_dir / "metric_max_error.png"
    __create_metric_plot(epochs, metrics_history['max_error'], 'Max Error',
                      'Maximum Error', 'red', path)
    
    path = saving_dir / "metric_evs.png"
    __create_metric_plot(epochs, metrics_history['evs'], 'EVS',
                      'Explained Variance Score', 'teal', path,
                      reference_line={'y': 1.0, 'label': 'Perfect Score'})
    
    path = saving_dir / "metric_mbe.png"
    __create_metric_plot(epochs, metrics_history['mbe'], 'MBE',
                      'Mean Bias Error', 'navy', path,
                      reference_line={'y': 0, 'label': 'Zero Bias'})
    
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, learning_rates, color='magenta', linewidth=2.5,
           marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = saving_dir / "metric_learning_rate.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close("all")
    gc.collect()
    
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
    path = saving_dir / "metric_epoch_time.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close("all")
    gc.collect()
    __create_metrics_summary_file(
        metrics_history,
        train_losses,
        val_losses,
        epoch_times,
        learning_rates,
        saving_dir,
        split_config_path=split_config_path,
    )
