import os
import logging
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    confusion_matrix,
)
import mlflow
import mlflow.pytorch

warnings.filterwarnings("ignore", message="Can't initialize NVML")
logger = logging.getLogger(__name__)


class LSTMSequenceClassifier(nn.Module):
    """LSTM-based classifier for per-timestep sequence classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
        )

        fc_in = self.hidden_size * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(fc_in, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for per-timestep classification.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Logits of shape (batch, seq_len, num_classes)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        num_directions = 2 if self.bidirectional else 1

        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=x.device,
        )
        c0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=x.device,
        )

        out, _ = self.lstm(x, (h0, c0))
        out_reshaped = out.reshape(-1, out.size(2))
        logits_reshaped = self.fc(out_reshaped)
        logits = logits_reshaped.reshape(batch_size, seq_len, self.num_classes)

        return logits

    def _compute_metrics(
        self, y_true: list, y_pred: list
    ) -> Tuple[float, float, float, float, float]:
        """Compute classification metrics."""
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        iou = jaccard_score(y_true, y_pred, average="weighted", zero_division=0)
        return acc, prec, rec, f1, iou

    def _process_batch(
        self, batch_data: tuple, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and move batch data to device."""
        if len(batch_data) == 3:
            X_batch, y_batch, _ = batch_data
        else:
            X_batch, y_batch = batch_data

        return X_batch.to(device), y_batch.to(device)

    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        epoch: int,
        num_epochs: int,
    ) -> Tuple[float, list, list]:
        """Execute one training epoch."""
        self.train()
        train_loss = 0.0
        y_true_all, y_pred_all = [], []
        total_samples = 0

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
            colour="green",
        )

        for batch_data in batch_bar:
            X_batch, y_batch = self._process_batch(batch_data, device)

            optimizer.zero_grad()
            outputs = self(X_batch)

            outputs_flat = outputs.view(-1, self.num_classes)
            y_batch_flat = y_batch.view(-1)

            loss = criterion(outputs_flat, y_batch_flat)
            loss.backward()
            optimizer.step()

            valid_mask = y_batch_flat != -1
            num_valid = valid_mask.sum().item()
            total_samples += num_valid
            train_loss += loss.item() * num_valid

            preds = torch.argmax(outputs_flat, dim=1).detach().cpu().numpy()
            y_true = y_batch_flat.cpu().numpy()
            valid_indices = y_true != -1

            y_pred_all.extend(preds[valid_indices])
            y_true_all.extend(y_true[valid_indices])

            batch_bar.set_postfix(loss=loss.item())

        train_loss = train_loss / total_samples if total_samples > 0 else 0.0
        return train_loss, y_true_all, y_pred_all

    def _validate_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, list, list]:
        """Execute one validation epoch."""
        self.eval()
        val_loss = 0.0
        y_true_all, y_pred_all = [], []
        total_samples = 0

        with torch.no_grad():
            for batch_data in val_loader:
                X_val, y_val = self._process_batch(batch_data, device)
                outputs = self(X_val)

                outputs_flat = outputs.view(-1, self.num_classes)
                y_val_flat = y_val.view(-1)

                loss = criterion(outputs_flat, y_val_flat)

                valid_mask = y_val_flat != -1
                num_valid = valid_mask.sum().item()
                total_samples += num_valid
                val_loss += loss.item() * num_valid

                preds = torch.argmax(outputs_flat, dim=1).cpu().numpy()
                y_true = y_val_flat.cpu().numpy()
                valid_indices = y_true != -1

                y_pred_all.extend(preds[valid_indices])
                y_true_all.extend(y_true[valid_indices])

        val_loss = val_loss / total_samples if total_samples > 0 else 0.0
        return val_loss, y_true_all, y_pred_all

    def _save_confusion_matrix(
        self,
        y_true: list,
        y_pred: list,
        idx_to_label: dict,
        epoch: int,
        model_path: str,
    ) -> Optional[str]:
        """Generate and save confusion matrix."""
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                logger.warning(
                    f"Skipping confusion matrix at epoch {epoch + 1}: no valid labels."
                )
                return None

            class_indices = sorted(idx_to_label.keys())
            class_names = [idx_to_label[i] for i in class_indices]

            cm = confusion_matrix(y_true, y_pred, labels=class_indices)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Greens",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
            )

            ax.set_title(f"Confusion Matrix (Epoch {epoch + 1})")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

            fig_path = os.path.join(
                model_path, f"confusion_matrix_epoch_{epoch + 1}.png"
            )
            plt.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)

            return fig_path
        except Exception as e:
            logger.warning(f"Could not save confusion matrix at epoch {epoch + 1}: {e}")
            return None

    def save_experiment_summary(
        self,
        file_path: str,
        model_path: str,
        run_id: str,
        run_name: str,
        experiment_name: str,
        learning_rate: float,
        num_epochs: int,
        patience: int,
        train_metrics_history: Dict[str, list],
        val_metrics_history: Dict[str, list],
        scheduler_factor: float,
        scheduler_patience: int,
        min_lr: float,
        notes: Optional[str] = None,
    ):
        """Write comprehensive experiment summary to file."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        with open(file_path, "w") as f:
            f.write("===== EXPERIMENT SUMMARY =====\n")
            f.write(f"Timestamp (UTC): {timestamp}\n")
            f.write(f"Experiment Name: {experiment_name}\n")
            f.write(f"MLflow Run ID: {run_id}\n")
            f.write(f"Run Name: {run_name}\n")
            f.write(f"Model Folder: {model_path}\n\n")

            f.write("===== MODEL CONFIGURATION =====\n")
            f.write(f"Model class: {self.__class__.__name__}\n")
            f.write(f"Input size: {self.input_size}\n")
            f.write(f"Hidden size: {self.hidden_size}\n")
            f.write(f"Num layers: {self.num_layers}\n")
            f.write(f"Bidirectional: {self.bidirectional}\n")
            f.write(f"Dropout: {self.dropout}\n")
            f.write(f"Num classes: {self.num_classes}\n")
            f.write("Task: Per-timestep sequence classification\n\n")

            f.write("===== TRAINING PARAMETERS =====\n")
            f.write(f"Initial learning rate: {learning_rate}\n")
            f.write(f"Epochs completed: {num_epochs}\n")
            f.write(f"Early stopping patience: {patience}\n")
            f.write("Optimizer: AdamW\n")
            f.write("Loss: CrossEntropyLoss (ignore_index=-1)\n")
            f.write("Scheduler: ReduceLROnPlateau\n")
            f.write(f"Scheduler factor: {scheduler_factor}\n")
            f.write(f"Scheduler patience: {scheduler_patience}\n")
            f.write(f"Minimum learning rate: {min_lr}\n\n")

            f.write("===== MODEL ARCHITECTURE =====\n")
            f.write(f"{self}\n\n")

            if notes:
                f.write("===== NOTES =====\n")
                f.write(f"{notes}\n\n")

            f.write("===== EPOCH-BY-EPOCH METRICS =====\n\n")
            epochs = len(train_metrics_history["loss"])
            for e in range(epochs):
                f.write(f"Epoch {e + 1}\n")
                f.write(
                    f"  Train - loss: {train_metrics_history['loss'][e]:.6f}, "
                    f"acc: {train_metrics_history['acc'][e]:.6f}, "
                    f"precision: {train_metrics_history['precision'][e]:.6f}, "
                    f"recall: {train_metrics_history['recall'][e]:.6f}, "
                    f"f1: {train_metrics_history['f1'][e]:.6f}, "
                    f"iou: {train_metrics_history['iou'][e]:.6f}, "
                    f"lr: {train_metrics_history['lr'][e]:.8f}\n"
                )
                f.write(
                    f"  Val   - loss: {val_metrics_history['loss'][e]:.6f}, "
                    f"acc: {val_metrics_history['acc'][e]:.6f}, "
                    f"precision: {val_metrics_history['precision'][e]:.6f}, "
                    f"recall: {val_metrics_history['recall'][e]:.6f}, "
                    f"f1: {val_metrics_history['f1'][e]:.6f}, "
                    f"iou: {val_metrics_history['iou'][e]:.6f}\n\n"
                )

            f.write("===== END OF SUMMARY =====\n")

        logger.info(f"Experiment summary saved to {file_path}")

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        patience: int = 10,
        device: Optional[torch.device] = None,
        idx_to_label: Optional[dict] = None,
        model_path: str = "models",
        run_name: str = "LSTM_Training",
        experiment_name: str = "LSTM_Activity_Classifier",
        save_confusion_every: int = 5,
        notes: Optional[str] = None,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 3,
        min_lr: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Train the model with MLflow tracking, ReduceLROnPlateau scheduler,
        and early stopping.

        Returns:
            Dictionary containing training history, validation history, and run_id
        """
        os.makedirs(model_path, exist_ok=True)
        model_name = os.path.join(model_path, "activity_detector.pth")
        summary_path = os.path.join(model_path, "experiment_summary.txt")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"No device specified. Using device: {device}")

        self.to(device)
        logger.info(f"Device: {device}")
        logger.info(
            f"Training setup: {num_epochs} epochs, lr={learning_rate:.6f}, patience={patience}"
        )

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr,
        )

        best_val_loss = np.inf
        counter = 0

        train_history = {
            "loss": [],
            "acc": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "iou": [],
            "lr": [],
        }
        val_history = {
            "loss": [],
            "acc": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "iou": [],
        }

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            mlflow.log_params(
                {
                    "input_size": self.input_size,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "num_classes": self.num_classes,
                    "learning_rate": learning_rate,
                    "bidirectional": self.bidirectional,
                    "dropout": self.dropout,
                    "patience": patience,
                    "optimizer": "AdamW",
                    "loss": "CrossEntropyLoss",
                    "task": "per_timestep_classification",
                    "scheduler": "ReduceLROnPlateau",
                    "scheduler_factor": scheduler_factor,
                    "scheduler_patience": scheduler_patience,
                    "min_lr": min_lr,
                }
            )

            epoch_bar = tqdm(
                range(num_epochs), desc="Training", leave=True, colour="blue"
            )

            for epoch in epoch_bar:
                train_loss, y_train_true, y_train_pred = self._train_epoch(
                    train_loader, criterion, optimizer, device, epoch, num_epochs
                )
                train_acc, train_prec, train_rec, train_f1, train_iou = (
                    self._compute_metrics(y_train_true, y_train_pred)
                )

                val_loss, y_val_true, y_val_pred = self._validate_epoch(
                    val_loader, criterion, device
                )
                val_acc, val_prec, val_rec, val_f1, val_iou = self._compute_metrics(
                    y_val_true, y_val_pred
                )

                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]["lr"]

                train_history["loss"].append(train_loss)
                train_history["acc"].append(train_acc)
                train_history["precision"].append(train_prec)
                train_history["recall"].append(train_rec)
                train_history["f1"].append(train_f1)
                train_history["iou"].append(train_iou)
                train_history["lr"].append(current_lr)

                val_history["loss"].append(val_loss)
                val_history["acc"].append(val_acc)
                val_history["precision"].append(val_prec)
                val_history["recall"].append(val_rec)
                val_history["f1"].append(val_f1)
                val_history["iou"].append(val_iou)

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "train_precision": train_prec,
                        "val_precision": val_prec,
                        "train_recall": train_rec,
                        "val_recall": val_rec,
                        "train_f1": train_f1,
                        "val_f1": val_f1,
                        "train_iou": train_iou,
                        "val_iou": val_iou,
                        "learning_rate_current": current_lr,
                    },
                    step=epoch,
                )

                if idx_to_label and (
                    (epoch % save_confusion_every == 0) or (epoch == num_epochs - 1)
                ):
                    fig_path = self._save_confusion_matrix(
                        y_val_true, y_val_pred, idx_to_label, epoch, model_path
                    )
                    if fig_path:
                        mlflow.log_artifact(fig_path)

                epoch_bar.set_postfix(
                    {
                        "train_loss": f"{train_loss:.4f}",
                        "val_loss": f"{val_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "train_acc": f"{train_acc:.3f}",
                        "val_acc": f"{val_acc:.3f}",
                        "train_f1": f"{train_f1:.3f}",
                        "val_f1": f"{val_f1:.3f}",
                        "train_iou": f"{train_iou:.3f}",
                        "val_iou": f"{val_iou:.3f}",
                    }
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    torch.save(self.state_dict(), model_name)
                else:
                    counter += 1
                    if counter >= patience:
                        logger.info("Early stopping triggered.")
                        break

            if os.path.exists(model_name):
                try:
                    best_state = torch.load(model_name, map_location=device)
                    self.load_state_dict(best_state)
                    logger.info(f"Training complete. Best model loaded from: {model_name}")
                except TypeError:
                    best_state = torch.load(model_name, map_location=device)
                    self.load_state_dict(best_state)
                    logger.info(f"Training complete. Best model loaded from: {model_name}")

            try:
                mlflow.pytorch.log_model(self, artifact_path="model")
            except Exception as e:
                logger.error(f"Failed to log model to MLflow: {e}")

            try:
                self.save_experiment_summary(
                    file_path=summary_path,
                    model_path=model_path,
                    run_id=run_id,
                    run_name=run_name,
                    experiment_name=experiment_name,
                    learning_rate=learning_rate,
                    num_epochs=len(train_history["loss"]),
                    patience=patience,
                    train_metrics_history=train_history,
                    val_metrics_history=val_history,
                    scheduler_factor=scheduler_factor,
                    scheduler_patience=scheduler_patience,
                    min_lr=min_lr,
                    notes=notes,
                )
                mlflow.log_artifact(summary_path)
            except Exception as e:
                logger.error(f"Failed to save experiment summary: {e}")

            logger.info(f"MLflow run completed. Run ID: {run_id}")

        return {
            "train_history": train_history,
            "val_history": val_history,
            "run_id": run_id,
        }

    def evaluate_loader(
        self,
        data_loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Evaluate the model on a data loader and return metrics."""
        if device is None:
            device = next(self.parameters()).device

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        loss, y_true, y_pred = self._validate_epoch(data_loader, criterion, device)
        acc, prec, rec, f1, iou = self._compute_metrics(y_true, y_pred)

        return {
            "loss": float(loss),
            "acc": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "iou": float(iou),
            "y_true": y_true,
            "y_pred": y_pred,
        }