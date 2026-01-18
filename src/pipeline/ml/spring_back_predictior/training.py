import numpy as np
import pandas as pd

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
from src.pipeline.ml.spring_back_predictior.plot_lstm import (
    plot_predictions,
    plot_training_history,
    plot_residuals,
    plot_true_vs_pred,
)

from src.pipeline.ml.spring_back_predictior.plot_random_forest import (
    plot_feature_importance_random_forest,
    plot_true_vs_pred_random_forest,
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
# Training Function
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
):
    # ------------------
    # Setup
    # ------------------
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epochs = training_params["max_epochs"]
    patience = training_params["stop_early_patience"]
    gradient_clip = training_params["gradient_clip"]
    verbose_every = training_params["verbose_every"]

    target_norm = TargetNormalizer(
        springbacks_train.mean().item(),
        springbacks_train.std().item(),
    )

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
        # ===== Training =====
        model.train()
        train_loss = 0.0
        train_true, train_pred = [], []

        for x, _, s in train_loader:
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

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        val_true, val_pred = [], []

        with torch.no_grad():
            for x, _, s in val_loader:
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

        # ===== Metrics =====
        train_r2 = r2_score(train_true, train_pred)
        val_r2 = r2_score(val_true, val_pred)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_r2"].append(train_r2)
        history["val_r2"].append(val_r2)
        history["train_rmse"].append(
            np.sqrt(mean_squared_error(train_true, train_pred))
        )
        history["val_rmse"].append(np.sqrt(mean_squared_error(val_true, val_pred)))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_loss)

        if (epoch + 1) % verbose_every == 0 or epoch == 0:
            epoch_pbar.set_postfix(
                train_loss=f"{train_loss:.3e}",
                val_loss=f"{val_loss:.3e}",
                val_r2=f"{val_r2:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        # ===== Early Stopping =====
        if val_loss < best_val_loss - training_params["stop_early_min_delta"]:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            epoch_pbar.close()
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # ------------------
    # Restore Best Model
    # ------------------
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # ------------------
    # Final Evaluation
    # ------------------
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, _, s in plot_loader:
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
        "y_true": y_true,
        "y_pred": y_pred,
    }

    plot_predictions(evaluation["y_true"], evaluation["y_pred"])
    plot_true_vs_pred(
        evaluation["y_true"],
        evaluation["y_pred"],
        evaluation["r2"],
    )
    plot_residuals(evaluation["y_true"], evaluation["y_pred"])
    plot_training_history(history)

    return model, history, evaluation


# -------------------- Main Training Function --------------------
def train_model_springback_random_forest(
    X_train, X_test, springbacks_train, springbacks_test
):
    # Convert to numpy
    X_tr = X_train.numpy()
    X_val = X_test.numpy()
    y_tr = springbacks_train.numpy().reshape(-1)
    y_val = springbacks_test.numpy().reshape(-1)

    n_samples, n_timesteps, n_features = X_tr.shape

    # -------------------- Flattened Model --------------------
    X_tr_flat = X_tr.reshape(n_samples, -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    rf_flat = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_flat.fit(X_tr_flat, y_tr)
    y_pred_flat = rf_flat.predict(X_val_flat)

    # Metrics for flattened model
    mse_flat = mean_squared_error(y_val, y_pred_flat)
    rmse_flat = np.sqrt(mse_flat)
    mae_flat = mean_absolute_error(y_val, y_pred_flat)
    r2_flat = r2_score(y_val, y_pred_flat)
    expl_var_flat = explained_variance_score(y_val, y_pred_flat)

    print("Random Forest (Flattened) Metrics:")
    print(
        f"MSE: {mse_flat:.4f}, RMSE: {rmse_flat:.4f}, MAE: {mae_flat:.4f}, R²: {r2_flat:.4f}, Explained Var: {expl_var_flat:.4f}"
    )

    # -------------------- Aggregated Model --------------------
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

    rf_agg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_agg.fit(X_tr_agg, y_tr)
    y_pred_agg = rf_agg.predict(X_val_agg)

    # Metrics for aggregated model
    mse_agg = mean_squared_error(y_val, y_pred_agg)
    rmse_agg = np.sqrt(mse_agg)
    mae_agg = mean_absolute_error(y_val, y_pred_agg)
    r2_agg = r2_score(y_val, y_pred_agg)
    expl_var_agg = explained_variance_score(y_val, y_pred_agg)

    print("\nRandom Forest (Aggregated) Metrics:")
    print(
        f"MSE: {mse_agg:.4f}, RMSE: {rmse_agg:.4f}, MAE: {mae_agg:.4f}, R²: {r2_agg:.4f}, Explained Var: {expl_var_agg:.4f}"
    )

    # -------------------- Plots --------------------
    plot_true_vs_pred_random_forest(y_val, y_pred_flat, y_pred_agg)

    # Feature importance
    feat_names_flat = [
        f"f{feat + 1}_t{t + 1}"
        for feat in range(n_features)
        for t in range(n_timesteps)
    ]
    feat_imp_flat = pd.DataFrame(
        {"feature": feat_names_flat, "importance": rf_flat.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    feat_names_agg = [
        f"f{feat + 1}_{agg}"
        for feat in range(n_features)
        for agg in ["mean", "std", "min", "max"]
    ]
    feat_imp_agg = pd.DataFrame(
        {"feature": feat_names_agg, "importance": rf_agg.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    print("\nTop 20 Flattened Features:")
    print(feat_imp_flat.head(20))
    print("\nTop 20 Aggregated Features:")
    print(feat_imp_agg.head(20))

    plot_feature_importance_random_forest(
        feat_imp_flat, title="RF Flattened Top 20 Features"
    )
    plot_feature_importance_random_forest(
        feat_imp_agg, title="RF Aggregated Top 20 Features"
    )
