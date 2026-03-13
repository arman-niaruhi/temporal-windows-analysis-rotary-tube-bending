import torch
import torch.nn as nn
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
import tempfile
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

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

from src.pipeline.ml.spring_back_predictior.plot_utils import (
    plot_prediction_difference_bars,
    plot_predictions_comparison,
    plot_true_vs_pred_scatter,
    plot_residuals_analysis,
    plot_metrics_comparison,
    plot_training_history,
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
# Model: TCN blocks
# ---------------------------------------------------------------------
class Chomp1d(nn.Module):
    """Removes right padding to keep causal length = input length."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding on the left via Conv1d padding + Chomp

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.act1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.act2(y)
        y = self.drop2(y)

        res = x if self.downsample is None else self.downsample(x)
        return self.out_act(y + res)


class TCN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        channels: Tuple[int, ...],
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = in_ch
        for i, ch in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_ch=prev,
                    out_ch=ch,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
            prev = ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return self.net(x)


# ---------------------------------------------------------------------
# Model: TCN -> (Bi)LSTM -> Head
# ---------------------------------------------------------------------
class TCNLSTMSpringback(nn.Module):
    """
    Hybrid model:
      [B,T,F] -> TCN over time (Conv1d) -> [B,T,C] -> (Bi)LSTM -> pooled -> MLP head
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tcn_channels: Tuple[int, ...] = (32, 64, 64),
        tcn_kernel_size: int = 5,
        tcn_dropout: float = 0.1,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        lstm_dropout: float = 0.0,
        bidirectional: bool = True,
        fc_dropout: float = 0.1,
        pool: str = "mean",  # "mean" or "last"
    ):
        super().__init__()
        self.pool = pool
        self.bidirectional = bidirectional

        self.tcn = TCN(
            in_ch=input_size,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
        )

        tcn_out = tcn_channels[-1]
        self.lstm = nn.LSTM(
            input_size=tcn_out,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden_size * (2 if bidirectional else 1)

        # Residual-ish head
        self.fc1 = nn.Linear(lstm_out_dim, lstm_out_dim)
        self.norm1 = nn.LayerNorm(lstm_out_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(fc_dropout)

        self.fc2 = nn.Linear(lstm_out_dim, lstm_out_dim // 2)
        self.norm2 = nn.LayerNorm(lstm_out_dim // 2)
        self.drop2 = nn.Dropout(fc_dropout)

        self.out = nn.Linear(lstm_out_dim // 2, output_size)

    @staticmethod
    def _mask_from_lengths(lengths: torch.Tensor, T: int) -> torch.Tensor:
        ar = torch.arange(T, device=lengths.device).unsqueeze(0)  # [1,T]
        return ar < lengths.unsqueeze(1)  # [B,T]

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [B,T,F] -> [B,F,T]
        x_conv = x.transpose(1, 2)
        x_conv = self.tcn(x_conv)  # [B,C,T]
        x_seq = x_conv.transpose(1, 2)  # [B,T,C]

        lstm_out, _ = self.lstm(x_seq)  # [B,T,H*dir]

        T = lstm_out.size(1)
        if mask is None and lengths is not None:
            mask = self._mask_from_lengths(lengths, T)

        if self.pool == "last":
            if lengths is not None:
                idx = (lengths - 1).clamp(min=0)  # [B]
                pooled = lstm_out[torch.arange(lstm_out.size(0), device=x.device), idx]
            else:
                pooled = lstm_out[:, -1, :]
        else:
            # masked mean pool
            if mask is None:
                pooled = lstm_out.mean(dim=1)
            else:
                mask_f = mask.to(lstm_out.dtype).unsqueeze(-1)  # [B,T,1]
                denom = mask_f.sum(dim=1).clamp(min=1.0)
                pooled = (lstm_out * mask_f).sum(dim=1) / denom

        h = self.fc1(pooled)
        h = self.act(h)
        h = self.norm1(h)
        h = self.drop1(h)
        h = h + pooled

        h = self.fc2(h)
        h = self.act(h)
        h = self.norm2(h)
        h = self.drop2(h)

        return self.out(h)


# ---------------------------------------------------------------------
# Training: TCN-LSTM
# ---------------------------------------------------------------------
def train_model_springback_tcn_lstm(
    seed: int,
    model_input_size: int,
    model_output_size: int,
    training_params: Dict[str, Any],
    springbacks_train: torch.Tensor,
    train_loader,
    val_loader,
    plot_loader,
    device: Optional[torch.device] = None,
    experiment_name: str = "Springback",
    model_name: str = "tcn_lstm",
) -> Tuple[nn.Module, Dict[str, list], Dict[str, float]]:
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"springback_{model_name}"):
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("model_name", model_name)

        set_seed(seed)

        if device is None:
            force_cpu = bool(training_params.get("force_cpu", False))
            device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

        n_epochs = int(training_params["max_epochs"])
        patience = int(training_params["stop_early_patience"])
        min_delta = float(training_params["stop_early_min_delta"])
        gradient_clip = float(training_params.get("gradient_clip", 0.0)) or None

        # target normalization
        target_norm = TargetNormalizer(
            springbacks_train.mean().item(),
            springbacks_train.std().item(),
        )

        # you can tune these; defaults are sensible
        tcn_channels = tuple(training_params.get("tcn_channels", [32, 64, 64]))
        tcn_kernel_size = int(training_params.get("tcn_kernel_size", 5))
        tcn_dropout = float(training_params.get("tcn_dropout", training_params.get("dropout", 0.1)))
        pool = training_params.get("pool", "mean")  # "mean" or "last"

        bidirectional = bool(training_params.get("bidirectional", True))

        loss_mae_weight = float(training_params.get("loss_mae_weight", 0.2))

        mlflow.log_params(
            {
                "model_name": model_name,
                "seed": seed,
                "input_size": model_input_size,
                "output_size": model_output_size,
                "tcn_channels": str(tcn_channels),
                "tcn_kernel_size": tcn_kernel_size,
                "tcn_dropout": tcn_dropout,
                "lstm_hidden_size": int(training_params["hidden_size"]),
                "lstm_num_layers": int(training_params["num_layers"]),
                "lstm_dropout": float(training_params["dropout"]),
                "bidirectional": bidirectional,
                "fc_dropout": float(training_params["fc_dropout"]),
                "pool": pool,
                "lr": float(training_params["lr"]),
                "weight_decay": float(training_params["weight_decay"]),
                "max_epochs": n_epochs,
                "early_stop_patience": patience,
                "early_stop_min_delta": min_delta,
                "gradient_clip": float(gradient_clip) if gradient_clip else 0.0,
                "scheduler_factor": float(training_params["schedular_factor"]),
                "scheduler_patience": int(training_params["schedular_patience"]),
                "loss_mae_weight": loss_mae_weight,
                "target_mean": target_norm.mean,
                "target_std": target_norm.std,
                "device": str(device),
            }
        )

        model = TCNLSTMSpringback(
            input_size=model_input_size,
            output_size=model_output_size,
            tcn_channels=tcn_channels,
            tcn_kernel_size=tcn_kernel_size,
            tcn_dropout=tcn_dropout,
            lstm_hidden_size=int(training_params["hidden_size"]),
            lstm_num_layers=int(training_params["num_layers"]),
            lstm_dropout=float(training_params["dropout"]),
            bidirectional=bidirectional,
            fc_dropout=float(training_params["fc_dropout"]),
            pool=pool,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_params["lr"]),
            weight_decay=float(training_params["weight_decay"]),
        )

        def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            mse = nn.functional.mse_loss(pred, target)
            mae = nn.functional.l1_loss(pred, target)
            return mse + loss_mae_weight * mae

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(training_params["schedular_factor"]),
            patience=int(training_params["schedular_patience"]),
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_r2": [],
            "val_r2": [],
            "train_rmse": [],
            "val_rmse": [],
            "train_mae": [],
            "val_mae": [],
            "lr": [],
        }

        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = -1
        patience_counter = 0

        def _parse_aux(aux, device_):
            lengths = None
            mask = None
            if torch.is_tensor(aux):
                if aux.dtype == torch.bool and aux.ndim == 2:
                    mask = aux.to(device_)
                elif aux.ndim == 1:
                    lengths = aux.to(device_)
            return lengths, mask

        # ------------------
        # Training loop
        # ------------------
        epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0
            train_true, train_pred = [], []

            for x, _, s, aux in train_loader:
                x = x.to(device).float()
                s = target_norm.normalize(s.to(device).float()).squeeze(-1)
                lengths, mask = _parse_aux(aux, device)

                optimizer.zero_grad(set_to_none=True)
                preds = model(x, lengths=lengths, mask=mask).squeeze(-1)
                loss = loss_fn(preds, s)
                loss.backward()

                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                optimizer.step()

                train_loss += loss.item() * x.size(0)
                train_true.append(s.detach().cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())

            train_loss /= len(train_loader.dataset)
            train_true_dn = target_norm.denormalize(np.concatenate(train_true))
            train_pred_dn = target_norm.denormalize(np.concatenate(train_pred))

            model.eval()
            val_loss = 0.0
            val_true, val_pred = [], []

            with torch.no_grad():
                for x, _, s, aux in val_loader:
                    x = x.to(device).float()
                    s = target_norm.normalize(s.to(device).float()).squeeze(-1)
                    lengths, mask = _parse_aux(aux, device)

                    preds = model(x, lengths=lengths, mask=mask).squeeze(-1)
                    loss = loss_fn(preds, s)

                    val_loss += loss.item() * x.size(0)
                    val_true.append(s.cpu().numpy())
                    val_pred.append(preds.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_true_dn = target_norm.denormalize(np.concatenate(val_true))
            val_pred_dn = target_norm.denormalize(np.concatenate(val_pred))

            train_r2 = r2_score(train_true_dn, train_pred_dn)
            val_r2 = r2_score(val_true_dn, val_pred_dn)

            tr_rmse = float(np.sqrt(mean_squared_error(train_true_dn, train_pred_dn)))
            va_rmse = float(np.sqrt(mean_squared_error(val_true_dn, val_pred_dn)))
            tr_mae = float(mean_absolute_error(train_true_dn, train_pred_dn))
            va_mae = float(mean_absolute_error(val_true_dn, val_pred_dn))
            lr = float(optimizer.param_groups[0]["lr"])

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["train_r2"].append(float(train_r2))
            history["val_r2"].append(float(val_r2))
            history["train_rmse"].append(tr_rmse)
            history["val_rmse"].append(va_rmse)
            history["train_mae"].append(tr_mae)
            history["val_mae"].append(va_mae)
            history["lr"].append(lr)

            epoch_pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                train_r2=f"{train_r2:.4f}",
                val_r2=f"{val_r2:.4f}",
                lr=f"{lr:.2e}",
            )

            mlflow.log_metrics(
                {
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "train_r2": float(train_r2),
                    "val_r2": float(val_r2),
                    "train_rmse": tr_rmse,
                    "val_rmse": va_rmse,
                    "train_mae": tr_mae,
                    "val_mae": va_mae,
                    "lr": lr,
                },
                step=epoch,
            )

            scheduler.step(val_loss)

            if val_loss < best_val_loss - min_delta:
                best_val_loss = float(val_loss)
                best_epoch = epoch
                best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

        mlflow.log_metric("best_val_loss", float(best_val_loss))
        mlflow.log_metric("best_epoch", int(best_epoch))

        # ------------------
        # Final eval on plot_loader
        # ------------------
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, _, s, aux in plot_loader:
                x = x.to(device).float()
                s = target_norm.normalize(s.to(device).float()).squeeze(-1)
                lengths, mask = _parse_aux(aux, device)

                preds = model(x, lengths=lengths, mask=mask).squeeze(-1)
                y_true.append(s.cpu().numpy())
                y_pred.append(preds.cpu().numpy())

        y_true_dn = target_norm.denormalize(np.concatenate(y_true))
        y_pred_dn = target_norm.denormalize(np.concatenate(y_pred))

        evaluation = {
            "r2": float(r2_score(y_true_dn, y_pred_dn)),
            "mae": float(mean_absolute_error(y_true_dn, y_pred_dn)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_dn, y_pred_dn))),
            "bias": float(np.mean(y_pred_dn - y_true_dn)),
        }

        mlflow.log_metrics(
            {
                "final_r2": evaluation["r2"],
                "final_mae": evaluation["mae"],
                "final_rmse": evaluation["rmse"],
                "final_bias": evaluation["bias"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_predictions_comparison(
                y_true_dn,
                y_pred_dn,
                model_name=model_name.upper().replace("_", "-"),
                save_path=os.path.join(tmpdir, "00_true_vs_pred_line.png"),
            )
            plot_true_vs_pred_scatter(
                y_true_dn,
                y_pred_dn,
                model_name=model_name.upper().replace("_", "-"),
                save_path=os.path.join(tmpdir, "01_true_vs_pred_scatter.png"),
            )
            plot_residuals_analysis(
                y_true_dn,
                y_pred_dn,
                model_name=model_name.upper().replace("_", "-"),
                save_path=os.path.join(tmpdir, "02_residuals_analysis.png"),
            )
            plot_prediction_difference_bars(
                y_true_dn,
                y_pred_dn,
                model_name=model_name.upper().replace("_", "-"),
                save_path=os.path.join(tmpdir, "03_residuals_bar.png"),
            )
            metrics_df = plot_metrics_comparison(
                y_true_dn,
                y_pred_dn,
                model_name=model_name.upper().replace("_", "-"),
                save_path=os.path.join(tmpdir, "04_metrics.png"),
            )
            metrics_df.to_csv(os.path.join(tmpdir, "metrics.csv"), index=False)

            plot_training_history(history, save_path=os.path.join(tmpdir, "05_training_history.png"))

            pd.DataFrame(
                {
                    "y_true": y_true_dn,
                    "y_pred": y_pred_dn,
                    "residual": y_pred_dn - y_true_dn,
                    "abs_error": np.abs(y_pred_dn - y_true_dn),
                }
            ).to_csv(os.path.join(tmpdir, f"{model_name}_predictions.csv"), index=False)

            mlflow.log_artifacts(tmpdir)

        mlflow.pytorch.log_model(model, f"{model_name}_model")
        return model, history, evaluation


# ---------------------------------------------------------------------
# Random Forest training (unchanged, but correct param logging)
# ---------------------------------------------------------------------
def train_model_springback_random_forest(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    springbacks_train: torch.Tensor,
    springbacks_test: torch.Tensor,
    sensor_names,
    experiment_name: str = "Springback",
):
    X_tr = X_train.detach().cpu().numpy()
    X_val = X_test.detach().cpu().numpy()
    y_tr = springbacks_train.detach().cpu().numpy().reshape(-1)
    y_val = springbacks_test.detach().cpu().numpy().reshape(-1)

    n_samples, n_timesteps, n_features = X_tr.shape
    assert len(sensor_names) == n_features, "sensor_names must match feature dimension"

    mlflow.set_experiment(experiment_name)

    def _run_rf_variant(
        variant_name: str,
        X_train_variant: np.ndarray,
        X_test_variant: np.ndarray,
        min_samples_leaf: Optional[int] = None,
    ):
        model_name = f"rf_{variant_name}"
        with mlflow.start_run(run_name=f"springback_{model_name}"):
            mlflow.set_tag("model_type", "rf")
            mlflow.set_tag("model_name", model_name)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("feature_variant", variant_name)
            mlflow.log_param("n_timesteps", int(n_timesteps))
            mlflow.log_param("n_features", int(n_features))
            mlflow.log_param("n_train_samples", int(X_train_variant.shape[0]))
            mlflow.log_param("n_test_samples", int(X_test_variant.shape[0]))

            rf_kwargs = {
                "n_estimators": 500,
                "max_depth": None,
                "random_state": 42,
                "n_jobs": -1,
                "bootstrap": True,
                "verbose": 1,
            }
            if min_samples_leaf is not None:
                rf_kwargs["min_samples_leaf"] = min_samples_leaf

            model = RandomForestRegressor(**rf_kwargs)
            model.fit(X_train_variant, y_tr)
            y_pred = model.predict(X_test_variant)

            metrics = {
                "mse": float(mean_squared_error(y_val, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
                "mae": float(mean_absolute_error(y_val, y_pred)),
                "r2": float(r2_score(y_val, y_pred)),
                "expl_var": float(explained_variance_score(y_val, y_pred)),
            }

            mlflow.log_params(
                {
                    "rf_n_estimators": model.n_estimators,
                    "rf_max_depth": model.max_depth,
                    "rf_min_samples_leaf": model.min_samples_leaf,
                }
            )
            mlflow.log_metrics(metrics)

            with tempfile.TemporaryDirectory() as tmpdir:
                display_name = model_name.upper().replace("_", "-")

                plot_predictions_comparison(
                    y_val,
                    y_pred,
                    model_name=display_name,
                    save_path=os.path.join(tmpdir, "00_true_vs_pred_line.png"),
                )
                plot_true_vs_pred_scatter(
                    y_val,
                    y_pred,
                    model_name=display_name,
                    save_path=os.path.join(tmpdir, "01_true_vs_pred_scatter.png"),
                )
                plot_residuals_analysis(
                    y_val,
                    y_pred,
                    model_name=display_name,
                    save_path=os.path.join(tmpdir, "02_residuals_analysis.png"),
                )
                plot_prediction_difference_bars(
                    y_val,
                    y_pred,
                    model_name=display_name,
                    save_path=os.path.join(tmpdir, "03_residuals_bar.png"),
                )
                plot_metrics_comparison(
                    y_val,
                    y_pred,
                    model_name=display_name,
                    save_path=os.path.join(tmpdir, "04_metrics.png"),
                )

                pd.DataFrame(
                    {
                        "y_true": y_val,
                        "y_pred": y_pred,
                        "residual": y_pred - y_val,
                        "abs_error": np.abs(y_pred - y_val),
                    }
                ).to_csv(
                    os.path.join(tmpdir, f"{model_name}_predictions.csv"), index=False
                )
                pd.DataFrame([metrics]).to_csv(
                    os.path.join(tmpdir, f"{model_name}_metrics.csv"), index=False
                )
                mlflow.log_artifacts(tmpdir)

            mlflow.sklearn.log_model(model, f"{model_name}_model")
            return model

    # Flattened RF run
    X_tr_flat = X_tr.reshape(n_samples, -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    rf_flat = _run_rf_variant("flat", X_tr_flat, X_val_flat, min_samples_leaf=2)

    # Aggregated RF run
    def aggregate_features(X: np.ndarray) -> np.ndarray:
        blocks = []
        for feat in range(X.shape[2]):
            d = X[:, :, feat]
            blocks.append(d.mean(axis=1))
            blocks.append(d.std(axis=1))
            blocks.append(d.min(axis=1))
            blocks.append(d.max(axis=1))
        return np.column_stack(blocks)

    X_tr_agg = aggregate_features(X_tr)
    X_val_agg = aggregate_features(X_val)
    rf_agg = _run_rf_variant("agg", X_tr_agg, X_val_agg)

    return rf_flat, rf_agg
