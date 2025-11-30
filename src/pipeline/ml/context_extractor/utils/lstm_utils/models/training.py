import time
import random
import shutil
import warnings
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import mlflow
import mlflow.pytorch

from src.pipeline.ml.context_extractor.utils.lstm_utils.models.att_lstm import (
    AttentionLSTM,
)
from src.pipeline.ml.context_extractor.utils.lstm_utils.data.data_preprocessor import (
    ProcessDataset,
)
from src.pipeline.ml.context_extractor.utils.lstm_utils.utils.metrics import (
    compute_all_metrics,
    compute_epoch_metrics,
)
from src.pipeline.ml.context_extractor.utils.lstm_utils.utils.visualization import (
    OrganizedImageSaver,
)
from src.pipeline.ml.context_extractor.utils.lstm_utils.utils.feature_importance import (
    analyze_feature_importance,
)


def move_images_to_mlflow_artifacts(image_saver):
    """
    Move entire image folder to MLflow experiment artifacts directory.
    Stores images in the same mlruns folder as the current run.
    """
    try:
        base_dir = image_saver.base_dir

        # Get the current MLflow run info
        run = mlflow.active_run()
        if run is None:
            print("✗ No active MLflow run")
            return None

        if base_dir.exists():
            mlflow.log_artifact(str(base_dir))
            shutil.rmtree(base_dir)
            return True
        else:
            print(f"✗ Images folder not found at {base_dir}")
            return None

    except Exception as e:
        print(f"✗ Error logging images to MLflow: {e}")
        return None


def save_experiment_description_as_text(EXPERIMENT_DESCRIPTION):
    """Save experiment description as a text file in MLflow artifacts"""
    desc_path = Path("experiment_description.txt")

    with open(desc_path, "w") as f:
        f.write(EXPERIMENT_DESCRIPTION)

    mlflow.log_artifact(str(desc_path))
    desc_path.unlink()  # Delete temporary file


def train_model(
    X,
    Y,
    params,
    sensor_names,
    target_feature_names,
    machine_part,
    preprocessing_info,
    annot_timesteps,
    mandrel_extraction_annot_timesteps
):
    if mlflow.active_run() is not None:
        mlflow.end_run()
        
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N_EXPERIMENTS_TRAIN_DATA, TIMESTEPS_IN_TRAIN_DATA, FEATURES_IN_TRAIN_DATA = X.shape
    N_CROSSCUT_TRAIN_DATA, PREDICTIONS_OUT_TRAIN_DATA, FEATURES_OUT_TRAIN_DATA = Y.shape

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )

    # -------------------------------
    # MLFLOW SETUP
    # -------------------------------
    mlflow.set_experiment(f"LSTM_Attention-{machine_part}")
    mlflow.set_tracking_uri("mlruns")

    EXPERIMENT_DESCRIPTION = f"""
    {machine_part} PART - LSTM Attention Model
    ============== PREPROCESSING INFO ====================
    {preprocessing_info}
    ==================== MODEL INFO ======================
    INPUT OF TARIN:
    N_EXPERIMENTS, N_TIMESTEPS_IN, N_FEATURES_IN = ({N_EXPERIMENTS_TRAIN_DATA}, {TIMESTEPS_IN_TRAIN_DATA}, {FEATURES_IN_TRAIN_DATA})
    OUTPUT_VALIDATION:
    N_CROSSCUT, PREDICTIONS_OUT, FEATURES_OUT = ({N_CROSSCUT_TRAIN_DATA}, {PREDICTIONS_OUT_TRAIN_DATA}, {FEATURES_OUT_TRAIN_DATA})
   
    GEOMETRY_FEATURES: {target_feature_names}
    ================== TRAINING INFO =====================
    {params}
    """

    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Create organized image directories
        image_saver = OrganizedImageSaver("images", machine_part=machine_part)
        save_experiment_description_as_text(EXPERIMENT_DESCRIPTION)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))

        # Compute global y-limits for plotting (using first feature)
        y_all = Y_val[:, :, 0].cpu().numpy()
        global_ymin, global_ymax = y_all.min(), y_all.max()
        margin = (global_ymax - global_ymin) * 0.1
        y_lim = (global_ymin - margin, global_ymax + margin)

        # Plotting batch
        val_ds = ProcessDataset(X_val, Y_val)
        plot_loader = DataLoader(val_ds, batch_size=min(64, len(val_ds)), shuffle=False)
        plot_X, plot_Y = next(iter(plot_loader))
        plot_X = plot_X.to(device)

        x_axis = np.arange(PREDICTIONS_OUT_TRAIN_DATA)
        n_samples = min(4, len(plot_Y))
        # Training setup
        train_ds = ProcessDataset(X_train, Y_train)
        train_loader = DataLoader(
            train_ds, batch_size=params["batch_size"], shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=32)

        # Model setup
        model = AttentionLSTM(
            input_features=FEATURES_IN_TRAIN_DATA,
            n_predictions=PREDICTIONS_OUT_TRAIN_DATA,
            output_features=FEATURES_OUT_TRAIN_DATA,
            hidden_dim=params["hidden_dim"],
            lstm_layers=params["lstm_layers"],
            dropout=params["dropout"],
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("trainable_parameters", trainable_params)

        optimizer = optim.AdamW(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        val_losses = []
        train_losses = []
        learning_rates = []
        
        # NEW: Store all metrics history
        metrics_history = {
            'mse': [], 'rmse': [], 'mae': [], 'r2': [], 
            'mape': [], 'max_error': [], 'evs': [], 'mbe': [], 'medae': []
        }
        
        best_val_loss = float("inf")
        best_state = None
        patience = 0
        epoch_times = []

        fpbar = tqdm(range(1, params["max_epochs"] + 1), desc="Training")
        for epoch in fpbar:
            epoch_start = time.time()

            # Training
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
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            val_preds_epoch = []
            val_targets_epoch = []

            with torch.no_grad():
                for Xb, Yb in val_loader:
                    Xb, Yb = Xb.to(device), Yb.to(device)
                    pred, _ = model(Xb)
                    val_loss += criterion(pred, Yb).item()

                    # Collect predictions and targets for metrics
                    val_preds_epoch.append(pred.cpu())
                    val_targets_epoch.append(Yb.cpu())

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Compute validation metrics
            val_preds_epoch = torch.cat(val_preds_epoch, dim=0)
            val_targets_epoch = torch.cat(val_targets_epoch, dim=0)
            metrics = compute_epoch_metrics(val_targets_epoch, val_preds_epoch)

            # NEW: Store metrics for plotting
            for key in metrics_history.keys():
                metrics_history[key].append(metrics[key])

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            learning_rates.append(current_lr)

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            # Log all metrics to MLflow
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

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = model.state_dict()
                patience = 0
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
            else:
                patience += 1

            # Save separate plots every 2 epochs
            if epoch % 2 == 0 or epoch == 1:
                with torch.no_grad():
                    pred, attn = model(plot_X)
                    pred_np = pred.cpu().numpy()
                    true_np = plot_Y.cpu().numpy()
                    attn_mean = attn.mean(0).cpu().numpy()

                idxs = random.sample(range(len(true_np)), min(n_samples, len(true_np)))

                # Prepare data for plotting
                pred_data = (true_np, pred_np, idxs)
                loss_data = (
                    list(range(1, len(val_losses) + 1)),
                    val_losses,
                    train_losses,
                )
                attn_data = attn_mean

                image_saver.save_epoch_plots(
                    X_train,
                    sensor_names,
                    target_feature_names,
                    pred_data,
                    loss_data,
                    attn_data,
                    epoch,
                    x_axis,
                    y_lim,
                    PREDICTIONS_OUT_TRAIN_DATA,
                    train_loss,
                    val_loss,
                    best_val_loss,
                    annot_timesteps,
                    mandrel_extraction_annot_timesteps
                )

            # Enhanced print statement with metrics
            fpbar.set_postfix(
                {
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
                },
                refresh=True,
            )
            if patience >= 10:
                mlflow.log_param("stopped_at_epoch", epoch)
                break

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation
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
        mlflow.pytorch.log_model(model.cpu(), "model")
        
        # ==================== PLOT ALL METRICS ====================
        print("\nPlotting comprehensive metrics...")
        plot_all_metrics(
            metrics_history=metrics_history,
            train_losses=train_losses,
            val_losses=val_losses,
            learning_rates=learning_rates,
            epoch_times=epoch_times,
            image_saver=image_saver,
        )
        
        # ==================== FEATURE IMPORTANCE ANALYSIS ====================
        print("\nStarting feature importance analysis...")

        # Perform comprehensive feature importance analysis
        combined_importance_df, all_importance_dfs, importance_paths = (
            analyze_feature_importance(
                model=model,
                X_val=X_val,
                val_loader=val_loader,
                feature_names=sensor_names,
                device=device,
            )
        )

        # Log feature importance to MLflow
        if combined_importance_df is not None:

            combined_csv_path = importance_paths.get("combined_csv")
            if combined_csv_path and Path(combined_csv_path).exists():
                mlflow.log_artifact(str(combined_csv_path))
            else:
                print("Warning: combined_csv path missing or does not exist; skipping MLflow log.")
            import pandas as pd
            # Log top 10 features
            combined_importance_df = {
                k: v if isinstance(v, list) else [v]
                for k, v in combined_importance_df.items()
            }
            combined_importance_df = pd.DataFrame(combined_importance_df)
            combined_importance_df.to_csv("feature_importance_summary.csv", index=False)
            mlflow.log_artifact("feature_importance_summary.csv")
            Path("feature_importance_summary.csv").unlink()  # Delete temporary file
        # Log images from the last epoch
        
        with torch.no_grad():
            model.eval()
            _, final_attn = model(plot_X)
            final_attn_mean = final_attn.mean(0).cpu().numpy()

        # Create the final line plot
        image_saver.plot_attention_lines_with_sensors(
            sensor_data=X_val,
            sensor_names=sensor_names,
            attn_mean=final_attn_mean,
            annot_timesteps=annot_timesteps,
            mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
            sample_idx=-1,  # Use last sample
        )
        move_images_to_mlflow_artifacts(image_saver)

        return {
            "model": model,
            "best_val_loss": best_val_loss,
            "final_metrics": final_metrics,
        }


def plot_all_metrics(metrics_history, train_losses, val_losses, learning_rates, epoch_times, image_saver):
    """
    Create individual plots for each training metric.
    """
    import matplotlib.pyplot as plt
    
    epochs = list(range(1, len(train_losses) + 1))
    saved_paths = []
    
    # 1. Loss curves (Train vs Val)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2.5, marker='o', markersize=4)
    ax.plot(epochs, val_losses, label='Val Loss', color='red', linewidth=2.5, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training and Validation Loss', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_loss.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path)
    print(f"✓ Saved Loss plot to {path}")
    
    # 2. MSE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_history['mse'], color='purple', linewidth=2.5, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('MSE', fontsize=14)
    ax.set_title('Mean Squared Error', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_mse.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path)
    print(f"✓ Saved MSE plot to {path}")
    
    # 3. RMSE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_history['rmse'], color='darkviolet', linewidth=2.5, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)
    ax.set_title('Root Mean Squared Error', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_rmse.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path)
    print(f"✓ Saved RMSE plot to {path}")
    
    # 4. MAE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_history['mae'], color='orange', linewidth=2.5, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('MAE', fontsize=14)
    ax.set_title('Mean Absolute Error', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_mae.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path)
    print(f"✓ Saved MAE plot to {path}")
    
    # 5. MedAE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_history['medae'], color='darkorange', linewidth=2.5, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('MedAE', fontsize=14)
    ax.set_title('Median Absolute Error', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_medae.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path)
    print(f"✓ Saved MedAE plot to {path}")
    
    # 6. R² Score
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_history['r2'], color='green', linewidth=2.5, marker='o', markersize=4)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Score')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('R² Score', fontsize=14)
    ax.set_title('R² Score (Coefficient of Determination)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_r2.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path)
    print(f"✓ Saved R² plot to {path}")
    
    # 7. MAPE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_history['mape'], color='brown', linewidth=2.5, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('MAPE (%)', fontsize=14)
    ax.set_title('Mean Absolute Percentage Error', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_mape.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path)
    print(f"✓ Saved MAPE plot to {path}")
    
    # 8. Max Error
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_history['max_error'], color='red', linewidth=2.5, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Max Error', fontsize=14)
    ax.set_title('Maximum Error', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_max_error.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path)
    print(f"✓ Saved Max Error plot to {path}")
    
    # 9. EVS (Explained Variance Score)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_history['evs'], color='teal', linewidth=2.5, marker='o', markersize=4)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Score')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('EVS', fontsize=14)
    ax.set_title('Explained Variance Score', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_evs.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path)
    print(f"✓ Saved EVS plot to {path}")
    
    # 10. MBE (Mean Bias Error)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_history['mbe'], color='navy', linewidth=2.5, marker='o', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Zero Bias')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('MBE', fontsize=14)
    ax.set_title('Mean Bias Error', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_mbe.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path)
    print(f"✓ Saved MBE plot to {path}")
    
    # 11. Learning Rate
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, learning_rates, color='magenta', linewidth=2.5, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_learning_rate.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path)
    print(f"✓ Saved Learning Rate plot to {path}")
    
    # 12. Epoch Times
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, epoch_times, color='cyan', linewidth=2.5, marker='o', markersize=4)
    ax.axhline(y=np.mean(epoch_times), color='red', linestyle='--', alpha=0.5, linewidth=2, 
               label=f'Avg: {np.mean(epoch_times):.2f}s')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_title('Training Time per Epoch', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = image_saver.base_dir / "metric_epoch_time.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path)
    print(f"✓ Saved Epoch Time plot to {path}")
    
    # 13. Summary text file
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
    
    saved_paths.append(summary_path)
    print(f"✓ Saved metrics summary to {summary_path}")
    
    # Log all artifacts to MLflow
    for path in saved_paths:
        mlflow.log_artifact(str(path))
    
    print(f"\n✓ Total of {len(saved_paths)} metric files saved and logged to MLflow")