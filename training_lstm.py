from src.pipeline.ml.context_extractor.utils.lstm_utils.lstm_preprocessing_utils import (
    LSTMPreprocessor
)

import json
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from src.pipeline.ml.context_extractor.utils.lstm_utils.feature_importance import analyze_feature_importance

import shutil
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import random
import warnings
import time
import mlflow
import mlflow.pytorch
from datetime import datetime

# ...existing code...
import os
# ...existing code...

def enforce_reproducibility(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For deterministic CUDA workspace (if supported)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import random as _rand

    _rand.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
# ...existing code...

class ProcessDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.float()
        self.Y = Y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class SimpleMLPAttention(nn.Module):
    def __init__(self, n_predictions, hidden_dim=128):
        super().__init__()
        self.n_predictions = n_predictions
        self.angle_embeddings = nn.Parameter(torch.randn(n_predictions, hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.xavier_uniform_(self.angle_embeddings)

    def forward(self, H):
        B, T, D = H.shape
        contexts, attns = [], []
        for a in range(self.n_predictions):
            scores = self.mlp(H + self.angle_embeddings[a]).squeeze(-1)
            w = torch.softmax(scores, dim=-1)
            ctx = (w.unsqueeze(-1) * H).sum(1)
            contexts.append(ctx)
            attns.append(w)
        return torch.stack(contexts, dim=1), torch.stack(attns, dim=1)


class AttentionLSTM(nn.Module):
    def __init__(
        self,
        input_features,
        n_predictions,
        output_features=1,
        hidden_dim=128,
        lstm_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        self.input_features = input_features
        self.n_predictions = n_predictions
        self.output_features = output_features
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_features,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.attention = SimpleMLPAttention(n_predictions, hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_features),
        )

    def forward(self, x):
        o, _ = self.lstm(x)
        o = self.ln(o)
        ctx, attn = self.attention(o)
        out = self.fc(ctx)
        return out, attn


def compute_epoch_metrics(y_true, y_pred):
    """Compute comprehensive metrics for a single epoch"""
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    # Flatten for overall metrics
    y_true_flat = y_true_np.flatten()
    y_pred_flat = y_pred_np.flatten()

    # Basic metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)

    # Additional metrics
    rmse = np.sqrt(mse)

    epsilon = 1e-8
    # Use mean of absolute values as denominator to avoid division by near-zero
    denominator = np.abs(y_true_flat) + epsilon
    # Alternative: use max(absolute_value, epsilon) for each element
    # denominator = np.maximum(np.abs(y_true_flat), epsilon)

    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / denominator)) * 100

    # Max Error
    max_error = np.max(np.abs(y_true_flat - y_pred_flat))

    # Explained Variance Score (similar to R² but different calculation)
    from sklearn.metrics import explained_variance_score

    evs = explained_variance_score(y_true_flat, y_pred_flat)

    # Mean Bias Error (shows if model systematically over/under predicts)
    mbe = np.mean(y_pred_flat - y_true_flat)

    # Median Absolute Error (more robust to outliers than MAE)
    from sklearn.metrics import median_absolute_error

    medae = median_absolute_error(y_true_flat, y_pred_flat)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "max_error": max_error,
        "evs": evs,
        "mbe": mbe,
        "medae": medae,
    }


def compute_all_metrics(y_true, y_pred):
    """Compute comprehensive metrics"""
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    mse = mean_squared_error(y_true_np.flatten(), y_pred_np.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_np.flatten(), y_pred_np.flatten())
    r2 = r2_score(y_true_np.flatten(), y_pred_np.flatten())

    per_pred_mse = np.mean((y_true_np - y_pred_np) ** 2, axis=(0, 2))
    per_pred_mae = np.mean(np.abs(y_true_np - y_pred_np), axis=(0, 2))

    if y_true_np.ndim == 3 and y_true_np.shape[2] > 1:
        per_feature_mse = np.mean((y_true_np - y_pred_np) ** 2, axis=(0, 1))
        per_feature_mae = np.mean(np.abs(y_true_np - y_pred_np), axis=(0, 1))
    else:
        per_feature_mse = None
        per_feature_mae = None

    max_error = np.max(np.abs(y_true_np - y_pred_np))
    mean_error = np.mean(y_pred_np - y_true_np)
    std_error = np.std(y_pred_np - y_true_np)
    residuals = y_pred_np - y_true_np

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "max_error": float(max_error),
        "mean_error": float(mean_error),
        "std_error": float(std_error),
        "per_prediction_mse": per_pred_mse.tolist(),
        "per_prediction_mae": per_pred_mae.tolist(),
        "residuals": residuals,
    }

    if per_feature_mse is not None:
        metrics["per_feature_mse"] = per_feature_mse.tolist()
        metrics["per_feature_mae"] = per_feature_mae.tolist()

    return metrics


class OrganizedImageSaver:
    def __init__(self, base_dir="images"):
        self.base_dir = Path(base_dir)

        # Create four main folders
        self.predictions_dir = self.base_dir / "01_predictions"
        self.loss_dir = self.base_dir / "02_loss"
        self.attention_dir = self.base_dir / "03_attention"
        self.attention_csv_dir = self.base_dir / "03_attention_csv"

        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.loss_dir.mkdir(parents=True, exist_ok=True)
        self.attention_dir.mkdir(parents=True, exist_ok=True)
        self.attention_csv_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_count = 0

    def save_epoch_plots(
        self,
        sensor_data,
        feature_names,
        output_feature_names,
        pred_data,
        loss_data,
        attn_data,
        epoch,
        x_axis,
        y_lim,
        PREDICTIONS_OUT,
        train_loss,
        val_loss,
        best_val_loss,
        annot_timesteps,
    ):
        """Save each subplot as a separate image in organized folders"""

        plt.style.use("tableau-colorblind10")

        true_np, pred_np, idxs = pred_data
        num_samples = len(idxs)
        n_features = true_np.shape[-1]

        # Horizontal layout: one row per sample, one column per feature
        nrows = num_samples
        ncols = n_features

        fig_pred, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5 * ncols, 3.5 * nrows),
            sharex=True,
            sharey=False
        )

        axes = np.array(axes).reshape(nrows, ncols)

        for row_i, idx in enumerate(idxs):
            for feat in range(n_features):
                ax = axes[row_i, feat]

                ax.plot(
                    x_axis,
                    true_np[idx, :, feat],
                    "o-",
                    lw=2.2,
                    ms=4,
                    label="True Value",
                )

                ax.plot(
                    x_axis,
                    pred_np[idx, :, feat],
                    "--s",
                    lw=1.8,
                    ms=4,
                    alpha=0.9,
                    label="Prediction",
                )

                ax.set_ylim(*y_lim)
                ax.grid(True, linestyle=":", alpha=0.55)

                # Row labels (left-most column)
                if feat == 0:
                    ax.set_ylabel(f"Sample {row_i}", fontsize=12, weight="bold")

                # Column labels (top row)
                if row_i == 0:
                    ax.set_title(output_feature_names[feat], fontsize=13, weight="bold")

                # Legend only once per row
                if feat == n_features - 1:
                    ax.legend(fontsize=9, loc="upper right")

        # Common labels
        fig_pred.suptitle(
            f"Predictions – Epoch {epoch} ({num_samples} samples × {n_features} features)",
            fontsize=16,
            weight="bold",
        )

        fig_pred.supxlabel(f"Prediction Index (Total: {PREDICTIONS_OUT})", fontsize=13)
        fig_pred.supylabel("Target Value", fontsize=13)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        pred_path = self.predictions_dir / f"predictions_epoch_{epoch:04d}.png"
        fig_pred.savefig(pred_path, dpi=180, bbox_inches="tight")
        plt.close(fig_pred)



        # 2. LOSS PLOT
        fig_loss = plt.figure(figsize=(10, 7))
        ax_loss = fig_loss.add_subplot(111)

        epochs_list, val_losses, train_losses = loss_data

        ax_loss.plot(
            epochs_list,
            train_losses,
            color="#1f77b4",
            lw=3,
            alpha=0.7,
            label="Train MSE",
        )
        ax_loss.plot(epochs_list, val_losses, color="#d62728", lw=3, label="Val MSE")
        ax_loss.plot(
            epochs_list,
            [best_val_loss] * len(epochs_list),
            color="green",
            lw=2.5,
            ls="--",
            label="Best Val MSE",
        )

        ax_loss.set_xlabel("Epoch", fontsize=12)
        ax_loss.set_ylabel("MSE", fontsize=12)
        ax_loss.set_title(
            f"Training Progress - Epoch {epoch}\nTrain: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f}",
            fontweight="bold",
            fontsize=14,
        )
        ax_loss.grid(alpha=0.3)
        ax_loss.legend(fontsize=10)
        plt.tight_layout()

        loss_path = self.loss_dir / f"loss_epoch_{epoch:04d}.png"
        fig_loss.savefig(loss_path, dpi=150, bbox_inches="tight")
        plt.close(fig_loss)

        # 3. ATTENTION HEATMAP
        attn_mean = attn_data
        attn_path = self.attention_dir / f"attention_epoch_{epoch:04d}.png"
        plot_selected_features_with_attn_heatmap(
            sensor_data, feature_names, attn_mean, attn_path, annot_timesteps
        )
        attn_df = pd.DataFrame(
            attn_mean,
            index=[f"Pred_{i}" for i in range(attn_mean.shape[0])],
            columns=[f"Time_{i}" for i in range(attn_mean.shape[1])],
        )

        csv_path = self.attention_csv_dir / f"attention_epoch_{epoch:04d}.csv"
        attn_df.to_csv(csv_path, float_format="%.6f")

        self.epoch_count = epoch

        return pred_path, loss_path, attn_path, csv_path


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


# def split_experiments(
#     sensor_df, target_df, machine_part, to_58_included, val_ratio=0.0, seed=42
# ):
#     """
#     Split data by experiment groups into train, validation, and test sets,
#     ensuring that experiment IDs from the same group are never split.

#     experiment_groups: list of lists, each containing experiment IDs from your JSON
#     """
#     with open("data/ml/unique_experiment_ids.json", "r") as f:
#         experiment_groups = json.load(f)
#     sensor_df = sensor_df.copy()
#     if machine_part == "DECLAMPING":
#         to_58_included = True
        
#     if to_58_included:
#         experiment_groups = [
#             sub for sub in experiment_groups if all(x >= 58 for x in sub)
#         ]

#     sensor_df["Experiment_ID"] = sensor_df["Experiment_ID"].astype(int)
#     target_df = target_df.copy()
#     target_df["Experiment_ID"] = target_df["Experiment_ID"].astype(int)
#     # Shuffle experiment groups as units
#     np.random.seed(seed)
#     np.random.shuffle(experiment_groups)

#     n_total_groups = len(experiment_groups)
#     n_val_groups = int(val_ratio * n_total_groups)

#     # Assign groups to splits
#     val_groups = experiment_groups[:n_val_groups]
#     train_groups = experiment_groups[n_val_groups:]

#     # Flatten experiment IDs for each split
#     train_exps = [eid for group in train_groups for eid in group]
#     val_exps = [eid for group in val_groups for eid in group]

#     # Filter DataFrame by experiment IDs
#     train_sensor_df = sensor_df[sensor_df["Experiment_ID"].isin(train_exps)].copy()
#     val_sensor_df = sensor_df[sensor_df["Experiment_ID"].isin(val_exps)].copy()
#     train_target_df = target_df[target_df["Experiment_ID"].isin(train_exps)].copy()
#     val_target_df = target_df[target_df["Experiment_ID"].isin(val_exps)].copy()
#     # Split data
    
#     return (
#         train_sensor_df,
#         val_sensor_df,
#         train_target_df,
#         val_target_df,
#         {"train": train_exps, "val": val_exps},
#     )


def save_experiment_description_as_text(EXPERIMENT_DESCRIPTION):
    """Save experiment description as a text file in MLflow artifacts"""
    desc_path = Path("experiment_description.txt")

    with open(desc_path, "w") as f:
        f.write(EXPERIMENT_DESCRIPTION)

    mlflow.log_artifact(str(desc_path))
    desc_path.unlink()  # Delete temporary file


def train_model(
 X,Y,
    params,
    sensor_names,
    target_feature_names,
    machine_part,
    preprocessing_info,
    annot_timesteps
):
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Seed for reproducibility
    
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N_EXPERIMENTS_TRAIN_DATA, TIMESTEPS_IN_TRAIN_DATA, FEATURES_IN_TRAIN_DATA = (
        X.shape
    )
    N_CROSSCUT_TRAIN_DATA, PREDICTIONS_OUT_TRAIN_DATA, FEATURES_OUT_TRAIN_DATA = (
        Y.shape
    )


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
        image_saver = OrganizedImageSaver("images")
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

        # Log model
        mlflow.pytorch.log_model(model, "model")

        # Add this at the end of train_model function, after final evaluation and before mlflow.log_metrics(metrics_to_log)

        # ==================== FEATURE IMPORTANCE ANALYSIS ====================
        print("\nStarting feature importance analysis...")

        # Perform comprehensive feature importance analysis
        combined_importance_df, all_importance_dfs, importance_paths = analyze_feature_importance(
            model=model,
            X_val=X_val,
            val_loader=val_loader,
            feature_names=sensor_names,
            device=device
        )

        # Log feature importance to MLflow
        if combined_importance_df is not None:
            # Log the combined importance CSV
            mlflow.log_artifact(str(importance_paths['combined_csv']))
            
            # Log top 10 features as parameters
            top_10_features = combined_importance_df.head(10)['Feature'].tolist()
            mlflow.log_param("top_10_features", ", ".join(top_10_features))
            
            # Log individual importance scores for top 5 features
            for i, row in combined_importance_df.head(5).iterrows():
                feature_name = row['Feature'].replace('_mean', '')
                mlflow.log_metric(f"importance_rank_{i+1}_{feature_name}", row['Average_Rank'])
            
            print(f"\nTop 10 Most Important Features:")
            print("="*80)
            for i, row in top_10_features.iterrows():
                print(f"{i+1}. {row['Feature']} (Average Importance: {row['Average_Rank']:.4f})")
            print("="*80)

        # Log all importance plots to MLflow artifacts
        if importance_paths:
            for path_name, path_value in importance_paths.items():
                if path_value and Path(path_value).exists():
                    try:
                        mlflow.log_artifact(str(path_value))
                        print(f"✓ Logged {path_name} to MLflow")
                    except Exception as e:
                        print(f"✗ Failed to log {path_name}: {e}")

        # Add feature importance results to return dictionary
        final_metrics['feature_importance'] = combined_importance_df
        final_metrics['importance_paths'] = importance_paths

        # Clean up temporary files
        if Path("images/04_feature_importance").exists():
            try:
                mlflow.log_artifact("images/04_feature_importance")
                shutil.rmtree("images/04_feature_importance")
                print("✓ Feature importance artifacts logged to MLflow")
            except Exception as e:
                print(f"✗ Error logging feature importance artifacts: {e}")

        print("\nFeature importance analysis complete!")
        # ======================================================================
        
        # Log images from the last epoch
        move_images_to_mlflow_artifacts(image_saver)

        return {
            "model": model,
            "best_val_loss": best_val_loss,
            "final_metrics": final_metrics,
        }


def plot_selected_features_with_attn_heatmap(
    sensor_data,
    sensor_names,
    attn_mean,
    attn_path,
    annot_timesteps=None,
    sample_idx=100,
    figsize=(25, 12),
):
    """
    Plots selected features with attention heatmap at the bottom.
    Includes legend for the top plot on the right side.
    Enhanced with beautiful styling and improved aesthetics.
    """
    # Set beautiful style parameters
    rcParams["font.family"] = "sans-serif"
    rcParams["font.size"] = 10

    # Remove '_mean' from all feature names
    cleaned_feature_names = [name.replace("_mean", "") for name in sensor_names]

    # Create figure with subplots
    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.clf()
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.25, wspace=0.3)
    ax_main = fig.add_subplot(gs[0])
    ax_heatmap = fig.add_subplot(gs[1])

    # Get the data for the selected sample
    sample_data = sensor_data[-1, :, :]
    main_timesteps = sample_data.shape[0]
    n_attention_heads = attn_mean.shape[0]
    attn_timesteps = attn_mean.shape[1]

    # --- MAIN PLOT WITH LEGEND ---
    # Use a sophisticated color palette
    colors = plt.cm.tab20(np.linspace(0, 1, len(cleaned_feature_names)))

    # Plot each feature with enhanced styling
    for i, (feature_name, color) in enumerate(zip(cleaned_feature_names, colors)):
        ax_main.plot(
            sample_data[:, i],
            color=color,
            linewidth=2.5,
            alpha=0.85,
            label=feature_name,
            marker="o",
            markersize=3,
            markevery=max(1, main_timesteps // 20),
        )  # Smart marker placement

    # Style main plot
    ax_main.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
    ax_main.set_ylabel("Feature Value", fontsize=12, fontweight="bold", labelpad=10)
    ax_main.grid(True, alpha=0.2, linestyle="--", linewidth=0.8, color="gray")
    ax_main.set_axisbelow(True)

    # Remove top and right spines for cleaner look
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["left"].set_linewidth(1.2)
    ax_main.spines["bottom"].set_linewidth(1.2)
    ax_main.spines["left"].set_color("#333333")
    ax_main.spines["bottom"].set_color("#333333")

    if annot_timesteps and (machine_part == "COMPLETE"):
        annot_labels = [
            "Start-Clamping",
            "Start-Bending",
            "Start-Declamping",
            "End-Clamping",
        ]  # Optional short labels

        for ts, label in zip(annot_timesteps, annot_labels):
            # Vertical line for visibility
            ax_main.axvline(ts, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

            # Annotated text placed slightly above the data region
            ax_main.annotate(
                label,
                xy=(ts, sample_data[:, :].max()),  # anchor at top of plot
                xytext=(0, 10),  # offset upward
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8),
            )

    ax_main.set_xlim(0, main_timesteps - 1)
    ax_main.set_facecolor("#f9f9f9")
    ax_main.set_title("Sensor Data Over Time", fontsize=14, fontweight="bold", pad=15)

    # Add legend with enhanced styling
    legend = ax_main.legend(
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

    # --- ATTENTION HEATMAP ---
    # Ensure heatmap has exactly the same number of timesteps as main plot
    if attn_timesteps != main_timesteps:
        print(f"Resizing attention from {attn_timesteps} to {main_timesteps} timesteps")
        attn_data_resized = np.zeros((n_attention_heads, main_timesteps))
        for i in range(n_attention_heads):
            x_original = np.arange(attn_timesteps)
            x_target = np.linspace(0, attn_timesteps - 1, main_timesteps)
            attn_data_resized[i] = np.interp(x_target, x_original, attn_mean[i])
        attn_data = attn_data_resized
    else:
        attn_data = attn_mean

    # Create enhanced heatmap
    im = ax_heatmap.imshow(
        attn_data,
        aspect="auto",
        cmap="magma",  # More visually appealing colormap
        interpolation="bilinear",
        extent=[0, main_timesteps - 1, 0, n_attention_heads - 1],
    )

    # Style heatmap
    ax_heatmap.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
    ax_heatmap.set_ylabel("Attention Head", fontsize=9, fontweight="bold", labelpad=10)

    ax_heatmap.set_yticks(np.arange(n_attention_heads))
    # Reverse the label order
    ax_heatmap.set_yticklabels(
        [f"{i + 1}" for i in reversed(range(n_attention_heads))], fontsize=5
    )

    ax_heatmap.set_xlim(0, main_timesteps - 1)
    ax_heatmap.set_facecolor("white")
    ax_heatmap.set_title(
        "Attention Head Intensity", fontsize=14, fontweight="bold", pad=15
    )

    # Remove spines for cleaner look
    ax_heatmap.spines["top"].set_visible(False)
    ax_heatmap.spines["right"].set_visible(False)
    ax_heatmap.spines["left"].set_linewidth(1.2)
    ax_heatmap.spines["bottom"].set_linewidth(1.2)
    ax_heatmap.spines["left"].set_color("#333333")
    ax_heatmap.spines["bottom"].set_color("#333333")

    # Add colorbar with enhanced styling
    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.9, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=11, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(1.2)

    # Fine-tune layout
    plt.tight_layout()

    # Get positions for alignment
    pos_main = ax_main.get_position()
    pos_heat = ax_heatmap.get_position()

    # Make heatmap width match main plot width
    ax_heatmap.set_position([pos_heat.x0, pos_heat.y0, pos_main.width, pos_heat.height])

    # Reposition colorbar to align properly
    cbar.ax.set_position(
        [pos_main.x0 + pos_main.width + 0.02, pos_heat.y0, 0.015, pos_heat.height]
    )

    fig.savefig(attn_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def resample_experiment_fast(group, n=46, agg_metric="mean"):
    """
    Optimized resampling function using vectorized operations.
    Up to 10-100x faster than the original implementation.
    """
    # Sort by time
    group = group.sort_values("Time_[s]")
    time_col = group["Time_[s]"].values

    # Assign each row to a time bin
    time_bins = np.linspace(time_col.min(), time_col.max(), n + 1)
    bin_indices = np.digitize(time_col, time_bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n - 1)

    # Get experiment ID
    exp_id = group["Experiment_ID"].iloc[0]

    # Select numeric columns only
    cols_to_process = [
        col for col in group.columns if col not in ["Time_[s]", "Experiment_ID"]
    ]

    results = []

    # Process each bin
    for bin_idx in range(n):
        mask = bin_indices == bin_idx
        if not mask.any():
            continue

        row_data = {"Experiment_ID": exp_id}

        for col in cols_to_process:
            values = group[col].values[mask]
            if len(values) == 0:
                continue

            # Compute metric using vectorized operations
            if agg_metric == "mean":
                row_data[f"{col}_mean"] = values.mean()
            elif agg_metric == "median":
                row_data[f"{col}_median"] = np.median(values)
            elif agg_metric == "min":
                row_data[f"{col}_min"] = values.min()
            elif agg_metric == "max":
                row_data[f"{col}_max"] = values.max()
            elif agg_metric == "range":
                row_data[f"{col}_range"] = values.ptp()
            elif agg_metric == "std":
                row_data[f"{col}_std"] = values.std()
            elif agg_metric == "var":
                row_data[f"{col}_var"] = values.var()
            elif agg_metric == "mad":
                row_data[f"{col}_mad"] = np.abs(values - values.mean()).mean()
            elif agg_metric == "rms":
                row_data[f"{col}_rms"] = np.sqrt((values**2).mean())
            elif agg_metric == "skew":
                row_data[f"{col}_skew"] = skew(values)
            elif agg_metric == "kurtosis":
                row_data[f"{col}_kurtosis"] = kurtosis(values)
            elif agg_metric == "energy":
                row_data[f"{col}_energy"] = (values**2).sum()
            elif agg_metric == "entropy":
                abs_vals = np.abs(values)
                probs = abs_vals / (abs_vals.sum() + 1e-12)
                row_data[f"{col}_entropy"] = entropy(probs + 1e-12)

        results.append(row_data)

    return pd.DataFrame(results)


# Alternative: Ultra-fast version using pandas groupby (even faster forfaltruese 'mean', 'std', 'min', 'max')
def resample_experiment_ultrafast(group, n=46, metric="mean"):
    """
    Ultra-optimized version using pandas groupby operations.
    Works best for basic metrics like mean, std, min, max, median.
    """
    group = group.sort_values("Time_[s]")
    time_col = group["Time_[s]"].values

    # Assign bins
    time_bins = np.linspace(time_col.min(), time_col.max(), n + 1)
    group["_bin"] = np.digitize(time_col, time_bins[:-1]) - 1
    group["_bin"] = group["_bin"].clip(0, n - 1)

    # Select columns to aggregate
    cols_to_agg = [
        col for col in group.columns if col not in ["Time_[s]", "Experiment_ID", "_bin"]
    ]

    # Map metric to pandas aggregation function
    agg_func_map = {
        "mean": "mean",
        "median": "median",
        "min": "min",
        "max": "max",
        "std": "std",
        "var": "var",
        "sum": "sum",
    }

    if metric in agg_func_map:
        # Use fast pandas groupby
        result = group.groupby("_bin")[cols_to_agg].agg(agg_func_map[metric])
        result = result.add_suffix(f"_{metric}")
        result["Experiment_ID"] = group["Experiment_ID"].iloc[0]
        return result.reset_index(drop=True)
    else:
        # Fall back to custom implementation
        return resample_experiment_fast(
            group.drop("_bin", axis=1), n, agg_metric=metric
        )


def prepare_data(input_path_param, preprocessing_param):
    def normalize_experiment(group, n=46):
        if len(group) > n:
            # Just take the first 46 rows
            return group.iloc[:n].copy()
        else:
            # Already 46 rows
            return group.copy()

    preprocessor = LSTMPreprocessor(
        sensors_path=input_path_param.get("sensors_path"),
        target_path=input_path_param.get("target_path"),
    )
    to_58_included = preprocessing_param.get("to_58_included", False)
    sensors_df, target_df = preprocessor.read_data()

    if input_path_param:
        target_df = target_df.groupby("Experiment_ID", group_keys=False).apply(
            normalize_experiment, n=46
        )
        target_df = target_df.reset_index(drop=True)
        sensors_df = sensors_df.reset_index()
        sensors_df = (
            sensors_df.groupby("Experiment_ID", group_keys=False)
            .apply(
                lambda g: resample_experiment_ultrafast(
                    g,
                    n=preprocessing_param.get("window_num", 40),
                    metric=preprocessing_param.get("agg_mertic", "mean"),
                )
            )
            .reset_index(drop=True)
        )

    
    columns = list(target_df.columns[1:])
    feature_idx_start, feature_idx_end = preprocessing_param.get("feature_indices")
    target_feature_names = columns[feature_idx_start:feature_idx_end]
    # train_sensor_df, val_sensor_df, train_target_df, val_target_df, experiment_ids = (
    #     split_experiments(
    #         sensors_df,
    #         target_df,
    #         input_path_param.get("machine_part"),
    #         preprocessing_param.get("to_58_excluded"),
    #         val_ratio=0.01
    #     )
    # )
 
    if machine_part == "DECLAMPING":
        to_58_included = True
        
    if to_58_included:
        # Subset the dataframes
        sensors_df = sensors_df[sensors_df["Experiment_ID"]>=58]
        target_df  = target_df[target_df["Experiment_ID"]>=58]

        
    X_train_numpy = preprocessor.group_and_pad(
        sensors_df, group_col="Experiment_ID"
    )
    Y_train_numpy = preprocessor.group_and_pad(
        target_df, group_col="Experiment_ID"
    )[:, :, feature_idx_start:feature_idx_end]


    X = torch.from_numpy(X_train_numpy).float()
    Y = torch.from_numpy(Y_train_numpy).float()
    sensor_names = list(sensors_df.columns[:-1])
    annot_timesteps = preprocessing_param.get("annot_timesteps", None)
    N = X.shape[1]
    annot_timesteps = [int((idx /1743) * N) for idx in annot_timesteps]
    return X, Y, sensor_names, target_feature_names, annot_timesteps


if __name__ == "__main__":
    with open("config/lstm_config.json", "r") as f:
        config = json.load(f)
    input_path_param = config.get("input_path_param")
    preprocessing_param = config.get("preprocessing_param")
    machine_part = input_path_param.get("machine_part")

    X, Y, sensor_names, target_feature_names, annot_timesteps = prepare_data(
        input_path_param=input_path_param, preprocessing_param=preprocessing_param
    )
    
    enforce_reproducibility(seed=config.get("seed", 42))
    result = train_model(
        X,
        Y,
        config.get("training_param"),
        sensor_names,
        target_feature_names,
        machine_part,
        config.get("preprocessing_param"),
        annot_timesteps
    )
