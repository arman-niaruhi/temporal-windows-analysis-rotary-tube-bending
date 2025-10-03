from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchviz import make_dot
import os

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

    # --- Custom train method ---
    def train_model(self, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, patience=10,
                device=None, model_path="models"):
        """
        Custom training loop with tqdm progress bar, accuracy & precision logging, and early stopping.
        """
        os.makedirs(model_path, exist_ok=True)
        
        model_name = f"{model_path}/Activity_Detector.joblib"
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        best_val_loss = np.inf
        counter = 0

        epoch_bar = tqdm(range(num_epochs), desc="Training", leave=True, colour="blue")
        for epoch in epoch_bar:
            # --- Training ---
            self.train()
            train_loss = 0.0
            y_train_true, y_train_pred = [], []

            batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, colour="green")
            for X_batch, y_batch in batch_bar:
                X_batch = X_batch.unsqueeze(1).to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = self(X_batch).squeeze(1)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                y_train_pred.extend(preds)
                y_train_true.extend(y_batch.cpu().numpy())

                # update batch bar
                batch_bar.set_postfix(loss=loss.item())

            train_loss /= len(train_loader.dataset)
            train_acc = accuracy_score(y_train_true, y_train_pred)
            train_prec = precision_score(y_train_true, y_train_pred, average="weighted", zero_division=0)

            # --- Validation ---
            self.eval()
            val_loss = 0.0
            y_val_true, y_val_pred = [], []

            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.unsqueeze(1).to(device)
                    y_val = y_val.to(device)
                    outputs = self(X_val).squeeze(1)
                    loss = criterion(outputs, y_val)
                    val_loss += loss.item() * X_val.size(0)

                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    y_val_pred.extend(preds)
                    y_val_true.extend(y_val.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_acc = accuracy_score(y_val_true, y_val_pred)
            val_prec = precision_score(y_val_true, y_val_pred, average="weighted", zero_division=0)

            # --- Update tqdm epoch bar ---
            lr = optimizer.param_groups[0]["lr"]
            epoch_bar.set_postfix({
                "lr": f"{lr:.2e}",
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "train_acc": f"{train_acc:.3f}",
                "val_acc": f"{val_acc:.3f}",
                "train_prec": f"{train_prec:.3f}",
                "val_prec": f"{val_prec:.3f}"
            })

            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                joblib.dump(self.state_dict(), model_name)
            else:
                counter += 1
                if counter >= patience:
                    print("\n⏹Early stopping triggered!")
                    break

        # --- Load best model ---
        best_state = joblib.load(model_name)
        self.load_state_dict(best_state)
        print("\nTraining complete. Best model loaded.")
        
    # --- Visualization function ---
    def visualize_model(self, input_size=10, seq_len=5, batch_size=2, out_file="lstm_model_graph"):
        """
        Generate a torchviz visualization of the model.
        Saves both SVG and PNG files.
        """

        x = torch.randn(batch_size, seq_len, input_size)
        y = self(x)

        dot = make_dot(y, params=dict(self.named_parameters()))
        dot.format = "png"
        png_path = dot.render(out_file, cleanup=True)
        print(f"Model graph saved at {png_path}")
        return png_path

