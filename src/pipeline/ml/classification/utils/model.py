import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm.auto import tqdm
import torch.nn as nn
import torch
from torch import optim
import joblib
from torchviz import make_dot
import warnings
from datetime import datetime
import json

warnings.filterwarnings("ignore", message="Can't initialize NVML")


class LSTMSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        bidirectional=False,
        dropout=0.2,
    ):
        """
        LSTM-based classifier for per-timestep classification.
        Predicts a label for each timestep in the sequence.
        """
        super(LSTMSequenceClassifier, self).__init__()

        # store config for later export / logging
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

    def forward(self, x, mask=None):
        """
        x expected shape: (batch, seq_len, input_size)
        mask optional shape: (batch, seq_len) - 1 for valid timesteps, 0 for padding
        returns logits of shape: (batch, seq_len, num_classes) for per-timestep prediction
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        num_directions = 2 if self.bidirectional else 1
        
        # LSTM initial states
        h0 = torch.zeros(
            self.num_layers * num_directions, batch_size, self.hidden_size, device=x.device
        )
        c0 = torch.zeros(
            self.num_layers * num_directions, batch_size, self.hidden_size, device=x.device
        )

        # LSTM forward - process all timesteps
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch, seq_len, hidden_size * num_directions)
        
        # Apply FC layer to each timestep
        # Reshape to apply FC: (batch * seq_len, hidden_size * num_directions)
        out_reshaped = out.reshape(-1, out.size(2))
        logits_reshaped = self.fc(out_reshaped)  # (batch * seq_len, num_classes)
        
        # Reshape back to (batch, seq_len, num_classes)
        logits = logits_reshaped.reshape(batch_size, seq_len, self.num_classes)
        
        return logits

    def save_experiment_summary(
        self,
        file_path,
        model_path,
        run_id,
        run_name,
        experiment_name,
        learning_rate,
        num_epochs,
        patience,
        train_metrics_history,
        val_metrics_history,
        notes=None,
    ):
        """
        Write a comprehensive experiment summary (.txt) to file_path.
        """
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
            f.write(f"Task: Per-timestep sequence classification\n\n")

            f.write("===== TRAINING PARAMETERS =====\n")
            f.write(f"Learning rate: {learning_rate}\n")
            f.write(f"Epochs requested: {num_epochs}\n")
            f.write(f"Early stopping patience: {patience}\n")
            f.write("Optimizer: Adam (default in code)\n")
            f.write("Loss: CrossEntropyLoss (ignore_index=-1 for padding)\n\n")

            f.write("===== MODEL ARCHITECTURE (print) =====\n")
            f.write(str(self))
            f.write("\n\n")

            if notes:
                f.write("===== NOTES =====\n")
                f.write(notes + "\n\n")

            f.write("===== EPOCH-BY-EPOCH METRICS =====\n\n")
            epochs = len(train_metrics_history["loss"])
            for e in range(epochs):
                f.write(f"Epoch {e + 1}\n")
                f.write(
                    f"  Train - loss: {train_metrics_history['loss'][e]:.6f}, acc: {train_metrics_history['acc'][e]:.6f}, "
                    f"precision: {train_metrics_history['precision'][e]:.6f}, recall: {train_metrics_history['recall'][e]:.6f}, f1: {train_metrics_history['f1'][e]:.6f}\n"
                )
                f.write(
                    f"  Val   - loss: {val_metrics_history['loss'][e]:.6f}, acc: {val_metrics_history['acc'][e]:.6f}, "
                    f"precision: {val_metrics_history['precision'][e]:.6f}, recall: {val_metrics_history['recall'][e]:.6f}, f1: {val_metrics_history['f1'][e]:.6f}\n\n"
                )

            f.write("===== END OF SUMMARY =====\n")

    def train_model(
        self,
        train_loader,
        val_loader,
        num_epochs=100,
        learning_rate=1e-3,
        patience=10,
        device=None,
        idx_to_label=None,
        model_path="models",
        run_name="LSTM_Training",
        experiment_name="LSTM_Activity_Classifier",
        save_confusion_every=5,
        notes=None,
    ):
        """
        Training loop for per-timestep classification with MLflow logging.
        Handles padded sequences with ignore_index=-1 in loss calculation.
        """

        os.makedirs(model_path, exist_ok=True)
        model_name = os.path.join(model_path, "activity_detector.pth")
        summary_path = os.path.join(model_path, "experiment_summary.txt")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Use ignore_index=-1 to ignore padded positions
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        best_val_loss = np.inf
        counter = 0

        # prepare metric histories
        train_history = {"loss": [], "acc": [], "precision": [], "recall": [], "f1": []}
        val_history = {"loss": [], "acc": [], "precision": [], "recall": [], "f1": []}

        # Start MLflow experiment tracking
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Log model + training params
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
                    "optimizer": "Adam",
                    "loss": "CrossEntropyLoss",
                    "task": "per_timestep_classification",
                }
            )

            epoch_bar = tqdm(range(num_epochs), desc="Training", leave=True, colour="blue")
            for epoch in epoch_bar:
                # --- Training ---
                self.train()
                train_loss = 0.0
                y_train_true, y_train_pred = [], []
                total_train_samples = 0

                batch_bar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{num_epochs}",
                    leave=False,
                    colour="green",
                )

                for batch_data in batch_bar:
                    # Handle both dataset types (with and without mask)
                    if len(batch_data) == 3:
                        X_batch, y_batch, mask_batch = batch_data
                        mask_batch = mask_batch.to(device)
                    else:
                        X_batch, y_batch = batch_data
                        mask_batch = None
                    
                    X_batch = X_batch.to(device)  # (batch, seq_len, input_size)
                    y_batch = y_batch.to(device)  # (batch, seq_len)

                    optimizer.zero_grad()
                    outputs = self(X_batch, mask=mask_batch)  # (batch, seq_len, num_classes)
                    
                    # Reshape for loss calculation
                    # outputs: (batch * seq_len, num_classes)
                    # y_batch: (batch * seq_len)
                    outputs_flat = outputs.view(-1, self.num_classes)
                    y_batch_flat = y_batch.view(-1)
                    
                    loss = criterion(outputs_flat, y_batch_flat)
                    loss.backward()
                    optimizer.step()

                    # Only count valid (non-padded) timesteps
                    valid_mask = y_batch_flat != -1
                    num_valid = valid_mask.sum().item()
                    total_train_samples += num_valid
                    train_loss += loss.item() * num_valid

                    # Get predictions only for valid timesteps
                    preds = torch.argmax(outputs_flat, dim=1).detach().cpu().numpy()
                    y_true = y_batch_flat.cpu().numpy()
                    
                    # Filter out padding (-1 labels)
                    valid_indices = y_true != -1
                    y_train_pred.extend(preds[valid_indices])
                    y_train_true.extend(y_true[valid_indices])
                    
                    batch_bar.set_postfix(loss=loss.item())

                train_loss /= total_train_samples
                train_acc = accuracy_score(y_train_true, y_train_pred)
                train_prec = precision_score(
                    y_train_true, y_train_pred, average="weighted", zero_division=0
                )
                train_rec = recall_score(
                    y_train_true, y_train_pred, average="weighted", zero_division=0
                )
                train_f1 = f1_score(
                    y_train_true, y_train_pred, average="weighted", zero_division=0
                )

                # --- Validation ---
                self.eval()
                val_loss = 0.0
                y_val_true, y_val_pred = [], []
                total_val_samples = 0

                with torch.no_grad():
                    for batch_data in val_loader:
                        # Handle both dataset types
                        if len(batch_data) == 3:
                            X_val, y_val, mask_val = batch_data
                            mask_val = mask_val.to(device)
                        else:
                            X_val, y_val = batch_data
                            mask_val = None
                        
                        X_val = X_val.to(device)
                        y_val = y_val.to(device)

                        outputs = self(X_val, mask=mask_val)
                        
                        outputs_flat = outputs.view(-1, self.num_classes)
                        y_val_flat = y_val.view(-1)
                        
                        loss = criterion(outputs_flat, y_val_flat)
                        
                        valid_mask = y_val_flat != -1
                        num_valid = valid_mask.sum().item()
                        total_val_samples += num_valid
                        val_loss += loss.item() * num_valid

                        preds = torch.argmax(outputs_flat, dim=1).cpu().numpy()
                        y_true = y_val_flat.cpu().numpy()
                        
                        valid_indices = y_true != -1
                        y_val_pred.extend(preds[valid_indices])
                        y_val_true.extend(y_true[valid_indices])

                val_loss /= total_val_samples
                val_acc = accuracy_score(y_val_true, y_val_pred)
                val_prec = precision_score(
                    y_val_true, y_val_pred, average="weighted", zero_division=0
                )
                val_rec = recall_score(
                    y_val_true, y_val_pred, average="weighted", zero_division=0
                )
                val_f1 = f1_score(
                    y_val_true, y_val_pred, average="weighted", zero_division=0
                )

                # Save histories
                train_history["loss"].append(train_loss)
                train_history["acc"].append(train_acc)
                train_history["precision"].append(train_prec)
                train_history["recall"].append(train_rec)
                train_history["f1"].append(train_f1)

                val_history["loss"].append(val_loss)
                val_history["acc"].append(val_acc)
                val_history["precision"].append(val_prec)
                val_history["recall"].append(val_rec)
                val_history["f1"].append(val_f1)

                # Log metrics to MLflow
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
                    },
                    step=epoch,
                )
                class_indices = sorted(idx_to_label.keys())
                class_names = [idx_to_label[i] for i in class_indices]

                # --- Confusion Matrix ---
                if (epoch % save_confusion_every == 0) or (epoch == num_epochs - 1):
                    try:
                        cm = confusion_matrix(
                            y_val_true,
                            y_val_pred,
                            labels=class_indices
                        )

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

                        mlflow.log_artifact(fig_path)

                    except Exception as e:
                        print(f"Warning: could not save confusion matrix at epoch {epoch+1}: {e}")

                # --- Update tqdm ---
                lr = optimizer.param_groups[0]["lr"]
                epoch_bar.set_postfix(
                    {
                        "lr": f"{lr:.2e}",
                        "train_loss": f"{train_loss:.4f}",
                        "val_loss": f"{val_loss:.4f}",
                        "train_acc": f"{train_acc:.3f}",
                        "val_acc": f"{val_acc:.3f}",
                        "train_f1": f"{train_f1:.3f}",
                        "val_f1": f"{val_f1:.3f}",
                    }
                )

                # --- Early stopping ---
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    torch.save(self.state_dict(), model_name)
                else:
                    counter += 1
                    if counter >= patience:
                        print("\n⏹ Early stopping triggered!")
                        break

            # --- Load best model ---
            if os.path.exists(model_name):
                best_state = torch.load(model_name, map_location=device)
                self.load_state_dict(best_state)

                print("\nTraining complete. Best model loaded from:", model_name)

            # Log model to MLflow
            try:
                mlflow.pytorch.log_model(self, artifact_path="model")
            except Exception as e:
                print("Warning: mlflow.pytorch.log_model failed:", e)

            # Save experiment summary
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
                    notes=notes,
                )
                mlflow.log_artifact(summary_path)
                print("Experiment summary saved to:", summary_path)
            except Exception as e:
                print("Warning: failed to write or log experiment summary:", e)

            print("Model and metrics logged to MLflow (run id:", run_id, ")")

        return {"train_history": train_history, "val_history": val_history, "run_id": run_id}

    def visualize_model(
        self, input_size=None, seq_len=5, batch_size=2, out_file="lstm_model_graph"
    ):
        """
        Generate a torchviz visualization of the model.
        """
        if input_size is None:
            input_size = self.input_size

        x = torch.randn(batch_size, seq_len, input_size)
        y = self(x)

        dot = make_dot(y, params=dict(self.named_parameters()))
        dot.format = "png"
        png_path = dot.render(out_file, cleanup=True)
        print(f"Model graph saved at {png_path}")
        return png_path