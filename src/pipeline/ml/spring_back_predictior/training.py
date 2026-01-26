import numpy as np
import pandas as pd
import os
import tempfile

import mlflow
import mlflow.sklearn
import mlflow.pytorch

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)

import torch
import torch.nn as nn
from typing import Optional, Dict
from tqdm.auto import tqdm

from src.pipeline.ml.spring_back_predictior.models import AttentionSpringbackLSTM
from src.pipeline.ml.spring_back_predictior.plot_utils import (
    plot_prediction_difference_bars,
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
class TargetNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std) + 1e-8

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# LSTM TRAINING WITH MLFLOW
# ---------------------------------------------------------------------
def train_model_springback_lstm(
    seed: int,
    model_input_size: int,
    model_output_size: int,
    training_params: Dict,
    springbacks_train: torch.Tensor,
    train_loader,
    val_loader,
    plot_loader,
    device: Optional[torch.device] = None,
    experiment_name: str = "Springback",
):

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # ------------------
        # Setup
        # ------------------
        set_seed(seed)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_epochs = training_params["max_epochs"]
        patience = training_params["stop_early_patience"]
        gradient_clip = training_params["gradient_clip"]

        target_norm = TargetNormalizer(
            springbacks_train.mean().item(),
            springbacks_train.std().item(),
        )

        # ------------------
        # Log Parameters
        # ------------------
        mlflow.log_params({
            "seed": seed,
            "input_size": model_input_size,
            "output_size": model_output_size,
            "hidden_size": training_params["hidden_size"],
            "num_layers": training_params["num_layers"],
            "dropout": training_params["dropout"],
            "fc_dropout": training_params["fc_dropout"],
            "lr": training_params["lr"],
            "weight_decay": training_params["weight_decay"],
            "max_epochs": n_epochs,
            "early_stop_patience": patience,
            "early_stop_min_delta": training_params["stop_early_min_delta"],
            "gradient_clip": gradient_clip,
            "scheduler_factor": training_params["schedular_factor"],
            "scheduler_patience": training_params["schedular_patience"],
        })

        # ------------------
        # Model
        # ------------------
        model = AttentionSpringbackLSTM(
            input_size=model_input_size,
            hidden_size=training_params["hidden_size"],
            num_layers=training_params["num_layers"],
            output_size=model_output_size,
            dropout=training_params["dropout"],
            fc_dropout=training_params["fc_dropout"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_params["lr"],
            weight_decay=training_params["weight_decay"],
        )

        def loss_fn(pred, target):
            mse = nn.functional.mse_loss(pred, target)
            mae = nn.functional.l1_loss(pred, target)
            return mse + 0.2 * mae

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=training_params["schedular_factor"],
            patience=training_params["schedular_patience"],
        )

        # ------------------
        # Tracking
        # ------------------
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_r2": [],
            "val_r2": [],
            "train_rmse": [],
            "val_rmse": [],
            "lr": [],
        }

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        # ------------------
        # Training Loop
        # ------------------
        epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0
            train_true, train_pred = [], []

            for x, _, s, _ in train_loader:
                x = x.to(device).float()
                s = target_norm.normalize(s.to(device).float()).squeeze(-1)

                optimizer.zero_grad()
                preds = model(x).squeeze(-1)
                loss = loss_fn(preds, s)
                loss.backward()

                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                optimizer.step()

                train_loss += loss.item() * x.size(0)
                train_true.append(s.detach().cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())

            train_loss /= len(train_loader.dataset)

            train_true = target_norm.denormalize(np.concatenate(train_true))
            train_pred = target_norm.denormalize(np.concatenate(train_pred))

            # ----- Validation -----
            model.eval()
            val_loss = 0.0
            val_true, val_pred = [], []

            with torch.no_grad():
                for x, _, s, _ in val_loader:
                    x = x.to(device).float()
                    s = target_norm.normalize(s.to(device).float()).squeeze(-1)
                    preds = model(x).squeeze(-1)
                    loss = loss_fn(preds, s)

                    val_loss += loss.item() * x.size(0)
                    val_true.append(s.cpu().numpy())
                    val_pred.append(preds.cpu().numpy())

            val_loss /= len(val_loader.dataset)

            val_true = target_norm.denormalize(np.concatenate(val_true))
            val_pred = target_norm.denormalize(np.concatenate(val_pred))

            train_r2 = r2_score(train_true, train_pred)
            val_r2 = r2_score(val_true, val_pred)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_r2"].append(train_r2)
            history["val_r2"].append(val_r2)
            history["train_rmse"].append(np.sqrt(mean_squared_error(train_true, train_pred)))
            history["val_rmse"].append(np.sqrt(mean_squared_error(val_true, val_pred)))
            history["lr"].append(optimizer.param_groups[0]["lr"])
            
            epoch_pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_r2=f"{val_r2:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_r2": train_r2,
                "val_r2": val_r2,
                "train_rmse": history["train_rmse"][-1],
                "val_rmse": history["val_rmse"][-1],
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            scheduler.step(val_loss)

            if val_loss < best_val_loss - training_params["stop_early_min_delta"]:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

        # ------------------
        # Final Evaluation
        # ------------------
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, _, s, _ in plot_loader:
                x = x.to(device).float()
                s = target_norm.normalize(s.to(device).float()).squeeze(-1)
                preds = model(x).squeeze(-1)
                y_true.append(s.cpu().numpy())
                y_pred.append(preds.cpu().numpy())

        y_true = target_norm.denormalize(np.concatenate(y_true))
        y_pred = target_norm.denormalize(np.concatenate(y_pred))

        evaluation = {
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "bias": float(np.mean(y_pred - y_true)),
        }

        mlflow.log_metrics({
            "final_r2": evaluation["r2"],
            "final_mae": evaluation["mae"],
            "final_rmse": evaluation["rmse"],
            "final_bias": evaluation["bias"],
        })

        # ------------------
        # Generate All Plots (using temp directory)
        # ------------------
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_prediction_difference_bars(
                y_true,
                y_pred,
                model_name="LSTM",
                save_path=os.path.join(tmpdir, "01_residuals_bar.png"),
            )
            
            
            # 6. Save predictions CSV
            pd.DataFrame({
                "y_true": y_true, 
                "y_pred": y_pred,
                "residual": y_pred - y_true,
                "abs_error": np.abs(y_pred - y_true)
            }).to_csv(os.path.join(tmpdir, "lstm_predictions.csv"), index=False)
            
            # Log all artifacts
            for file in os.listdir(tmpdir):
                mlflow.log_artifact(os.path.join(tmpdir, file))

        # Log model
        mlflow.pytorch.log_model(model, "lstm_model")
        return model, history, evaluation


# -------------------- Random Forest Training --------------------
def train_model_springback_random_forest(
    X_train, X_test, springbacks_train, springbacks_test, sensor_names, 
    experiment_name="Springback"
):
    # Convert tensors to numpy
    X_tr = X_train.numpy()
    X_val = X_test.numpy()
    y_tr = springbacks_train.numpy().reshape(-1)
    y_val = springbacks_test.numpy().reshape(-1)

    n_samples, n_timesteps, n_features = X_tr.shape

    # Start MLflow run
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        
        print(f"\n{'='*60}")
        print("Training Random Forest Models...")
        print(f"{'='*60}\n")
        
        # -------------------- Flattened Model --------------------
        print("Training RF Flattened Model...")
        X_tr_flat = X_tr.reshape(n_samples, -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        rf_flat = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
            bootstrap=True,
            verbose=1
        )

        rf_flat.fit(X_tr_flat, y_tr)
        y_pred_flat = rf_flat.predict(X_val_flat)

        # Metrics
        mse_flat = mean_squared_error(y_val, y_pred_flat)
        rmse_flat = np.sqrt(mse_flat)
        mae_flat = mean_absolute_error(y_val, y_pred_flat)
        r2_flat = r2_score(y_val, y_pred_flat)
        expl_var_flat = explained_variance_score(y_val, y_pred_flat)

        mlflow.log_params({
            "rf_flat_n_estimators": 500,
            "rf_flat_max_depth": "None",
            "rf_flat_min_samples_leaf": 2,
        })

        mlflow.log_metrics({
            "mse_flat": mse_flat,
            "rmse_flat": rmse_flat,
            "mae_flat": mae_flat,
            "r2_flat": r2_flat,
            "expl_var_flat": expl_var_flat
        })

        mlflow.sklearn.log_model(rf_flat, "rf_flat_model")
        
        print(f"RF Flattened - R²: {r2_flat:.6f}, RMSE: {rmse_flat:.6f}")

        # -------------------- Aggregated Model --------------------
        print("\nTraining RF Aggregated Model...")
        
        def aggregate_features(X):
            agg_features = []
            for feat in range(X.shape[2]):
                feat_data = X[:, :, feat]
                mean = feat_data.mean(axis=1)
                std = feat_data.std(axis=1)
                min_val = feat_data.min(axis=1)
                max_val = feat_data.max(axis=1)
                agg_features.extend([mean, std, min_val, max_val])
            return np.column_stack(agg_features)

        X_tr_agg = aggregate_features(X_tr)
        X_val_agg = aggregate_features(X_val)

        rf_agg = RandomForestRegressor(
            n_estimators=500, 
            max_depth=None, 
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        rf_agg.fit(X_tr_agg, y_tr)
        y_pred_agg = rf_agg.predict(X_val_agg)

        # Metrics
        mse_agg = mean_squared_error(y_val, y_pred_agg)
        rmse_agg = np.sqrt(mse_agg)
        mae_agg = mean_absolute_error(y_val, y_pred_agg)
        r2_agg = r2_score(y_val, y_pred_agg)
        expl_var_agg = explained_variance_score(y_val, y_pred_agg)

        mlflow.log_params({
            "rf_agg_n_estimators": 200,
            "rf_agg_max_depth": 10,
        })

        mlflow.log_metrics({
            "mse_agg": mse_agg,
            "rmse_agg": rmse_agg,
            "mae_agg": mae_agg,
            "r2_agg": r2_agg,
            "expl_var_agg": expl_var_agg
        })

        mlflow.sklearn.log_model(rf_agg, "rf_agg_model")
        
        print(f"RF Aggregated - R²: {r2_agg:.6f}, RMSE: {rmse_agg:.6f}")

        # -------------------- Generate All Plots --------------------
        print("\nGenerating plots...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_prediction_difference_bars(
                y_val,
                y_pred_flat,
                model_name="RF Flattened",
                save_path=os.path.join(tmpdir, "01_residuals_flattened.png"),
            )

            plot_prediction_difference_bars(
                y_val,
                y_pred_agg,
                model_name="RF Aggregated",
                save_path=os.path.join(tmpdir, "02_residuals_aggregated.png"),
            )

            feat_names_flat = [
                f"{sensor_names[feat]}_t{t + 1}"
                for feat in range(n_features)
                for t in range(n_timesteps)
            ]
            feat_imp_flat = pd.DataFrame({
                "feature": feat_names_flat,
                "importance": rf_flat.feature_importances_,
            }).sort_values(by="importance", ascending=False)

            feat_names_agg = [
                f"{sensor_names[feat]}_{agg}"
                for feat in range(n_features)
                for agg in ["mean", "std", "min", "max"]
            ]
            feat_imp_agg = pd.DataFrame({
                "feature": feat_names_agg,
                "importance": rf_agg.feature_importances_,
            }).sort_values(by="importance", ascending=False)

            feat_imp_flat.to_csv(os.path.join(tmpdir, "feat_imp_flat.csv"), index=False)
            feat_imp_agg.to_csv(os.path.join(tmpdir, "feat_imp_agg.csv"), index=False)
            
            # Save predictions CSVs
            pd.DataFrame({
                "y_true": y_val,
                "y_pred_flat": y_pred_flat,
                "y_pred_agg": y_pred_agg,
                "residual_flat": y_pred_flat - y_val,
                "residual_agg": y_pred_agg - y_val,
            }).to_csv(os.path.join(tmpdir, "rf_predictions.csv"), index=False)

            # Log all artifacts
            for file in os.listdir(tmpdir):
                mlflow.log_artifact(os.path.join(tmpdir, file))
        
        return rf_flat, rf_agg, feat_imp_flat, feat_imp_agg
