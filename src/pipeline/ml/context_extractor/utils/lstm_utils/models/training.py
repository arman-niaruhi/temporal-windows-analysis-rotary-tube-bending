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
        move_images_to_mlflow_artifacts(image_saver)

        return {
            "model": model,
            "best_val_loss": best_val_loss,
            "final_metrics": final_metrics,
        }
