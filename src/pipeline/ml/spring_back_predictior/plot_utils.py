import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# --------------------------------------------------
# 1. Predictions Over Sample Index (Line Plot)
# --------------------------------------------------
def plot_predictions_comparison(y_true, y_pred, model_name="Model", save_path=None, show=False):
    """
    Plot true vs predicted values as line plots over sample index.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values (can be dict with multiple models or single array)
        model_name: Name of the model or dict key
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    plt.figure(figsize=(14, 6))
    
    # Plot true values
    plt.plot(y_true, label="True", linewidth=2.5, alpha=0.9, color='black')
    
    # Handle multiple predictions
    if isinstance(y_pred, dict):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for idx, (name, pred) in enumerate(y_pred.items()):
            plt.plot(pred, label=name, linewidth=2, linestyle="--", 
                    alpha=0.8, color=colors[idx % len(colors)])
    else:
        plt.plot(y_pred, label=model_name, linewidth=2, linestyle="--", alpha=0.8)
    
    plt.title("Predicted vs True Springback Over Samples", fontsize=14, fontweight='bold')
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Springback", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


# --------------------------------------------------
# 1b. Prediction Difference Bar Chart
# --------------------------------------------------
def plot_prediction_difference_bars(
    y_true,
    y_pred,
    model_name="Model",
    save_path=None,
    show=False,
    max_bars=200,
):
    """
    Plot per-sample prediction differences (y_pred - y_true) as a bar chart.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Path to save the plot
        show: Whether to display the plot
        max_bars: Max number of bars to draw (samples are evenly downsampled)
    """
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    n_samples = len(residuals)

    if n_samples > max_bars:
        idx = np.linspace(0, n_samples - 1, max_bars, dtype=int)
        residuals = residuals[idx]
        x = idx
        title = f"{model_name} - Residuals (sampled {max_bars}/{n_samples})"
    else:
        x = np.arange(n_samples)
        title = f"{model_name} - Residuals (y_pred - y_true)"

    colors = np.where(residuals >= 0, "#3498db", "#e74c3c")
    plt.figure(figsize=(14, 6))
    plt.bar(x, residuals, color=colors, alpha=0.75, edgecolor="black", linewidth=0.3)
    plt.axhline(0, linestyle="--", linewidth=1.5, color="black")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Residual", fontsize=12)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# --------------------------------------------------
# 2. True vs Predicted Scatter Plot
# --------------------------------------------------
def plot_true_vs_pred_scatter(y_true, y_pred, model_name="Model", save_path=None, show=False):
    """
    Scatter plot of true vs predicted with ideal line and metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values (can be dict with multiple models or single array)
        model_name: Name of the model
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    # Handle multiple predictions
    if isinstance(y_pred, dict):
        n_models = len(y_pred)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, pred) in enumerate(y_pred.items()):
            ax = axes[idx]
            r2 = r2_score(y_true, pred)
            rmse = np.sqrt(mean_squared_error(y_true, pred))
            mae = mean_absolute_error(y_true, pred)
            
            ax.scatter(y_true, pred, alpha=0.5, s=40, edgecolor="none")
            
            min_val = min(y_true.min(), pred.min())
            max_val = max(y_true.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   linestyle="--", linewidth=2, color='red', label='Ideal')
            
            ax.set_xlabel("True Springback", fontsize=11)
            ax.set_ylabel("Predicted Springback", fontsize=11)
            ax.set_title(f"{name}\nR²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}", 
                        fontsize=11, fontweight='bold')
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
        
        plt.tight_layout()
    else:
        plt.figure(figsize=(7, 7))
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        plt.scatter(y_true, y_pred, alpha=0.5, s=40, edgecolor="none")
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                linestyle="--", linewidth=2, color='red', label='Ideal')
        
        plt.xlabel("True Springback", fontsize=12)
        plt.ylabel("Predicted Springback", fontsize=12)
        plt.title(f"{model_name} - True vs Predicted\nR²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}", 
                 fontsize=12, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


# --------------------------------------------------
# 3. Residual Analysis
# --------------------------------------------------
def plot_residuals_analysis(y_true, y_pred, model_name="Model", save_path=None, show=False):
    """
    Comprehensive residual analysis with multiple subplots.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values (can be dict with multiple models or single array)
        model_name: Name of the model
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if isinstance(y_pred, dict):
        n_models = len(y_pred)
        fig, axes = plt.subplots(n_models, 3, figsize=(18, 5*n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, pred) in enumerate(y_pred.items()):
            residuals = pred - y_true
            
            # Residuals vs Predicted
            axes[idx, 0].scatter(pred, residuals, alpha=0.5, s=30, edgecolor="none")
            axes[idx, 0].axhline(0, linestyle="--", linewidth=2, color='red')
            axes[idx, 0].set_xlabel("Predicted Springback", fontsize=11)
            axes[idx, 0].set_ylabel("Residual", fontsize=11)
            axes[idx, 0].set_title(f"{name} - Residuals vs Predicted", fontsize=11, fontweight='bold')
            axes[idx, 0].grid(True, linestyle="--", alpha=0.4)
            
            # Residual Distribution
            axes[idx, 1].hist(residuals, bins=40, alpha=0.7, edgecolor='black')
            axes[idx, 1].axvline(0, linestyle="--", linewidth=2, color='red')
            axes[idx, 1].set_xlabel("Residual", fontsize=11)
            axes[idx, 1].set_ylabel("Frequency", fontsize=11)
            axes[idx, 1].set_title(f"{name} - Residual Distribution", fontsize=11, fontweight='bold')
            axes[idx, 1].grid(True, linestyle="--", alpha=0.4)
            
            # Residuals vs Index
            axes[idx, 2].scatter(range(len(residuals)), residuals, alpha=0.5, s=30, edgecolor="none")
            axes[idx, 2].axhline(0, linestyle="--", linewidth=2, color='red')
            axes[idx, 2].set_xlabel("Sample Index", fontsize=11)
            axes[idx, 2].set_ylabel("Residual", fontsize=11)
            axes[idx, 2].set_title(f"{name} - Residuals vs Index", fontsize=11, fontweight='bold')
            axes[idx, 2].grid(True, linestyle="--", alpha=0.4)
        
        plt.tight_layout()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        residuals = y_pred - y_true
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=30, edgecolor="none")
        axes[0].axhline(0, linestyle="--", linewidth=2, color='red')
        axes[0].set_xlabel("Predicted Springback", fontsize=11)
        axes[0].set_ylabel("Residual", fontsize=11)
        axes[0].set_title(f"{model_name} - Residuals vs Predicted", fontsize=11, fontweight='bold')
        axes[0].grid(True, linestyle="--", alpha=0.4)
        
        # Residual Distribution
        axes[1].hist(residuals, bins=40, alpha=0.7, edgecolor='black')
        axes[1].axvline(0, linestyle="--", linewidth=2, color='red')
        axes[1].set_xlabel("Residual", fontsize=11)
        axes[1].set_ylabel("Frequency", fontsize=11)
        axes[1].set_title(f"{model_name} - Residual Distribution", fontsize=11, fontweight='bold')
        axes[1].grid(True, linestyle="--", alpha=0.4)
        
        # Residuals vs Index
        axes[2].scatter(range(len(residuals)), residuals, alpha=0.5, s=30, edgecolor="none")
        axes[2].axhline(0, linestyle="--", linewidth=2, color='red')
        axes[2].set_xlabel("Sample Index", fontsize=11)
        axes[2].set_ylabel("Residual", fontsize=11)
        axes[2].set_title(f"{model_name} - Residuals vs Index", fontsize=11, fontweight='bold')
        axes[2].grid(True, linestyle="--", alpha=0.4)
        
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


# --------------------------------------------------
# 4. Metrics Comparison Bar Chart
# --------------------------------------------------
def plot_metrics_comparison(y_true, y_pred, model_name='Model', save_path=None, show=False):
    """
    Compute and plot metrics for a single set of predictions.
    
    Args:
        y_true: Ground truth values (array-like)
        y_pred: Predicted values (array-like)
        model_name: Name of the model
        save_path: Path to save the plot
        show: Whether to display the plot
    Returns:
        df_metrics: DataFrame containing the metrics
    """
    
    # Compute metrics
    metrics_data = [{
        'Model': model_name,
        'R²': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'Bias': np.mean(y_pred - y_true)
    }]
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ['R²', 'RMSE', 'MAE', 'Bias']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    
    for metric, pos, color in zip(metrics, positions, colors):
        ax = axes[pos]
        bars = ax.bar(df_metrics['Model'], df_metrics[metric], color=color, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', linestyle="--", alpha=0.4)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
    return df_metrics


# --------------------------------------------------
# 5. Error Distribution Comparison
# --------------------------------------------------
def plot_error_distribution(y_true, y_pred_dict, save_path=None, show=False):
    """
    Compare error distributions across models.
    
    Args:
        y_true: Ground truth values
        y_pred_dict: Dictionary of {model_name: predictions}
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Absolute Error Distribution
    for idx, (name, pred) in enumerate(y_pred_dict.items()):
        abs_errors = np.abs(pred - y_true)
        axes[0].hist(abs_errors, bins=30, alpha=0.5, label=name, 
                    color=colors[idx % len(colors)], edgecolor='black')
    
    axes[0].set_xlabel("Absolute Error", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Absolute Error Distribution", fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)
    
    # Percentage Error Distribution
    for idx, (name, pred) in enumerate(y_pred_dict.items()):
        pct_errors = 100 * (pred - y_true) / (y_true + 1e-8)
        axes[1].hist(pct_errors, bins=30, alpha=0.5, label=name,
                    color=colors[idx % len(colors)], edgecolor='black')
    
    axes[1].set_xlabel("Percentage Error (%)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Percentage Error Distribution", fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


# --------------------------------------------------
# 6. Training History (for models with training curves)
# --------------------------------------------------
def plot_training_history(history, save_path=None, show=False):
    """
    Plot training history for LSTM or other iterative models.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_r2', 'val_r2', 'lr'
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2.5, color='#3498db')
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2.5, color='#e74c3c')
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)
    
    # R²
    axes[1].plot(epochs, history["train_r2"], label="Train R²", linewidth=2.5, color='#2ecc71')
    axes[1].plot(epochs, history["val_r2"], label="Val R²", linewidth=2.5, color='#f39c12')
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("R²", fontsize=12)
    axes[1].set_title("R² Score History", fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)
    
    # Learning Rate
    axes[2].plot(epochs, history["lr"], linewidth=2.5, color='#9b59b6')
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Learning Rate", fontsize=12)
    axes[2].set_title("Learning Rate Schedule", fontsize=13, fontweight='bold')
    axes[2].grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


# --------------------------------------------------
# 7. Feature Importance (for Random Forest)
# --------------------------------------------------
def plot_feature_importance(feat_imp_df, title="Feature Importance", top_n=20, save_path=None, show=False):
    """
    Plot feature importance for tree-based models.
    
    Args:
        feat_imp_df: DataFrame with 'feature' and 'importance' columns
        title: Plot title
        top_n: Number of top features to display
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 8))
    
    top_features = feat_imp_df.head(top_n)
    
    bars = plt.barh(top_features['feature'][::-1], top_features['importance'][::-1], 
                    color='#3498db', alpha=0.7, edgecolor='black')
    
    plt.xlabel("Importance", fontsize=12, fontweight='bold')
    plt.ylabel("Feature", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, axis='x', linestyle="--", alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


# --------------------------------------------------
# 8. Combined Summary Plot
# --------------------------------------------------
def plot_model_summary(y_true, y_pred, model_name="Model", history=None, save_path=None, show=False):
    """
    Create a comprehensive summary plot with all key visualizations.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        model_name: Name of the model
        history: Optional training history
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if history is not None:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    residuals = y_pred - y_true
    
    # 1. Predictions over samples
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(y_true, label="True", linewidth=2.5, alpha=0.9, color='black')
    ax1.plot(y_pred, label="Predicted", linewidth=2, linestyle="--", alpha=0.8, color='#3498db')
    ax1.set_xlabel("Sample Index", fontsize=11)
    ax1.set_ylabel("Springback", fontsize=11)
    ax1.set_title(f"{model_name} - Predictions vs True", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # 2. Scatter plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(y_true, y_pred, alpha=0.5, s=40, edgecolor="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    ax2.set_xlabel("True", fontsize=11)
    ax2.set_ylabel("Predicted", fontsize=11)
    ax2.set_title(f"R²={r2:.4f}", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    
    # 3. Residuals vs Predicted
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolor="none")
    ax3.axhline(0, linestyle="--", linewidth=2, color='red')
    ax3.set_xlabel("Predicted", fontsize=11)
    ax3.set_ylabel("Residual", fontsize=11)
    ax3.set_title("Residuals vs Predicted", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle="--", alpha=0.4)
    
    # 4. Residual distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(residuals, bins=40, alpha=0.7, edgecolor='black')
    ax4.axvline(0, linestyle="--", linewidth=2, color='red')
    ax4.set_xlabel("Residual", fontsize=11)
    ax4.set_ylabel("Frequency", fontsize=11)
    ax4.set_title("Residual Distribution", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle="--", alpha=0.4)
    
    # 5. Metrics summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    metrics_text = f"""
    MODEL METRICS
    {'='*30}
    R² Score:     {r2:.6f}
    RMSE:         {rmse:.6f}
    MAE:          {mae:.6f}
    Bias:         {np.mean(residuals):.6f}
    Std Dev:      {np.std(residuals):.6f}
    Max Error:    {np.max(np.abs(residuals)):.6f}
    {'='*30}
    """
    ax5.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 6-8. Training history (if available)
    if history is not None:
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.plot(epochs, history["train_loss"], label="Train", linewidth=2)
        ax6.plot(epochs, history["val_loss"], label="Val", linewidth=2)
        ax6.set_xlabel("Epoch", fontsize=11)
        ax6.set_ylabel("Loss", fontsize=11)
        ax6.set_title("Loss History", fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, linestyle="--", alpha=0.4)
        
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.plot(epochs, history["train_r2"], label="Train R²", linewidth=2)
        ax7.plot(epochs, history["val_r2"], label="Val R²", linewidth=2)
        ax7.set_xlabel("Epoch", fontsize=11)
        ax7.set_ylabel("R²", fontsize=11)
        ax7.set_title("R² History", fontsize=12, fontweight='bold')
        ax7.legend()
        ax7.grid(True, linestyle="--", alpha=0.4)
        
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.plot(epochs, history["lr"], linewidth=2, color='purple')
        ax8.set_xlabel("Epoch", fontsize=11)
        ax8.set_ylabel("Learning Rate", fontsize=11)
        ax8.set_title("LR Schedule", fontsize=12, fontweight='bold')
        ax8.grid(True, linestyle="--", alpha=0.4)
    
    plt.suptitle(f"{model_name} - Complete Analysis", fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
