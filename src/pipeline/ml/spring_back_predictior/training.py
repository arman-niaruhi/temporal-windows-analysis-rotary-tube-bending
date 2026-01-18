import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
from tqdm.auto import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.pipeline.ml.spring_back_predictior.models import AttentionSprigbackLSTM
from src.pipeline.ml.spring_back_predictior.plot import (
    plot_predictions,
    plot_training_history,
    plot_residuals,
    plot_true_vs_pred,
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
def train_model_springback(
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
    model = AttentionSprigbackLSTM(
        input_size=model_input_size,
        hidden_size=training_params["hidden_size"],
        num_layers=training_params["num_layers"],
        output_size=model_output_size,
        dropout=training_params["dropout"],
        fc_dropout=training_params["fc_dropout"],
        bidirectional=training_params["bidirectional"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_params["lr"],
        weight_decay=training_params["weight_decay"],
    )

    criterion = nn.SmoothL1Loss()

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
            loss = criterion(preds, s)
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
                loss = criterion(preds, s)

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
        history["val_rmse"].append(
            np.sqrt(mean_squared_error(val_true, val_pred))
        )
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
