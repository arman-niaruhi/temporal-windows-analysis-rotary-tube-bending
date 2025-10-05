import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import joblib
import os 

# -------------------------
# AngleAttentionLSTM Trainer
# -------------------------
class AngleAttentionLSTMTrainer:
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_angles, 
                 lr=1e-4, dropout=0.2, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.AngleAttentionLSTM(input_size, hidden_size, num_layers, output_size, num_angles, dropout)
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.num_angles = num_angles

    # -------------------------
    # Sinusoidal angle embedding
    # -------------------------
    @staticmethod
    def get_angle_embedding(angle_idx, hidden_dim, device):
        angle_idx = angle_idx.float().unsqueeze(1)  # (num_angles,1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, device=device) *
                             -(math.log(10000.0)/hidden_dim))
        emb = torch.zeros((angle_idx.size(0), hidden_dim), device=device)
        emb[:, 0::2] = torch.sin(angle_idx * div_term)
        emb[:, 1::2] = torch.cos(angle_idx * div_term)
        return emb

    # -------------------------
    # AngleAttentionLSTM model
    # -------------------------
    class AngleAttentionLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, num_angles, dropout=0.2):
            super().__init__()
            self.num_angles = num_angles
            self.hidden_size = hidden_size

            # LSTM encoder
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, bidirectional=True, dropout=dropout)

            # Attention projections
            self.Wk = nn.Linear(hidden_size*2, hidden_size*2)
            self.Wq = nn.Linear(hidden_size*2, hidden_size*2)

            # Output per angle
            self.fc = nn.ModuleList([nn.Linear(hidden_size*2, output_size) for _ in range(num_angles)])

        def forward(self, x):
            batch_size, seq_len, _ = x.size()
            lstm_out, _ = self.lstm(x)
            K = self.Wk(lstm_out)

            # Angle embeddings
            angle_indices = torch.arange(self.num_angles, device=x.device)
            angle_embs = AngleAttentionLSTMTrainer.get_angle_embedding(angle_indices, self.hidden_size*2, x.device)

            out_list = []
            attn_list = []

            for a in range(self.num_angles):
                q = self.Wq(angle_embs[a]).unsqueeze(0).expand(batch_size, -1)
                scores = torch.bmm(K, q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.hidden_size*2)
                attn_weights = F.softmax(scores, dim=1)
                attn_list.append(attn_weights)

                context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
                out_angle = self.fc[a](context)
                out_list.append(out_angle)

            out = torch.stack(out_list, dim=1)
            attn_weights_all = torch.stack(attn_list, dim=1)
            return out, attn_weights_all

        @staticmethod
        def most_attended_timestep(attn_weights):
            return attn_weights.argmax(dim=-1)

    # -------------------------
    # Training loop with early stopping
    # -------------------------
    def train(self, train_loader, test_loader=None, num_epochs=200, patience=10, save_path="best_model.pth"):
        best_loss = float('inf')
        counter = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                y_pred, _ = self.model(xb)
                loss = self.criterion(y_pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                running_loss += loss.item() * xb.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)

            # Optional test evaluation
            test_loss = None
            if test_loader is not None:
                self.model.eval()
                test_loss_total = 0.0
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        y_pred, _ = self.model(xb)
                        loss = self.criterion(y_pred, yb)
                        test_loss_total += loss.item() * xb.size(0)
                test_loss = test_loss_total / len(test_loader.dataset)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.6f}", end="")
            if test_loss is not None:
                print(f", Test Loss: {test_loss:.6f}")
            else:
                print()

            # Early stopping
            monitor_loss = test_loss if test_loss is not None else epoch_loss
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                counter = 0
                self.save(save_path)
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    # -------------------------
    # Save model
    # -------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)  # make sure folder exists
        joblib.dump(self.model, path)

    # -------------------------
    # Load model
    # -------------------------
    def load(self, path):
        joblib.load(path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {path}")

    # -------------------------
    # Predict
    # -------------------------
    def predict(self, X_tensor):
        self.model.eval()
        X_tensor = X_tensor.to(self.device)
        with torch.no_grad():
            Y_pred, attn_weights = self.model(X_tensor)
        return Y_pred.cpu().numpy(), attn_weights.cpu().numpy()
