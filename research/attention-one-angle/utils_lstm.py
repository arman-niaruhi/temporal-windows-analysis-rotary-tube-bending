from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_importance(X_sample, scores=None, target=None, attn_weights=None, top_n=3, target_angle = 1):
    if isinstance(X_sample, np.ndarray):
        X_sample = torch.tensor(X_sample, dtype=torch.float32)
    elif isinstance(X_sample, pd.DataFrame):
        X_sample = torch.tensor(X_sample.values, dtype=torch.float32)
    X_sample = X_sample.detach()

    num_features = X_sample.shape[1]
    seq_len = X_sample.shape[0]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_features)))

    for i in range(num_features):
        data = X_sample[:, i].numpy()
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        ax1.plot(
            norm_data + i,
            color=colors[i % len(colors)],
            linewidth=1.5,
            label=f"Feature {i}",
            alpha=0.8,
        )

    ax1.set_ylabel("Feature Value (Normalized + Offset)")
    ax1.set_title("All Features")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    if attn_weights is not None:
        attn_weights_np = attn_weights.numpy()
        max_idx = np.argmax(attn_weights_np)
        top_k = 5
        top_indices = np.argsort(attn_weights_np)[-top_k:]
        ax1.axvspan(
            top_indices[0],
            top_indices[-1],
            alpha=0.3,
            color="red",
            label="Highest Attention Area",
        )

    if attn_weights is not None:
        ax2.plot(attn_weights.numpy(), color="red", linewidth=2)
        ax2.set_title("Attention Weights Over Time")
        ax2.set_ylabel("Attention")
        ax2.grid(True, alpha=0.3)

    if target is not None:
        if isinstance(target, pd.DataFrame):
            targets_plot = target.values
        else:
            targets_plot = target
        ax3.plot(targets_plot, marker="o")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Target Value")
        ax3.set_title("Target Values")
        ax3.grid(True, alpha=0.3)
        ax3.axvspan(
            target_angle-0.5,
            target_angle+0.5,
            alpha=0.3,
            color="blue",
            label="Target Angle",
        )

    plt.tight_layout()
    plt.show()

    if scores is not None:
        top_indices = np.argsort([sc for _, _, sc in scores])[-top_n:][::-1]
        top_windows = [scores[i] for i in top_indices]
        
        
        
class LSTMAttentionModel(nn.Module):
    def __init__(
        self, input_size=17, hidden_size=32, num_layers=1, output_size=4, dropout=0.3  
    ):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = torch.softmax(self.attention(lstm_out), dim=1)      
        context = torch.sum(attn_scores * lstm_out, dim=1)                
        out = self.fc(context)   
        return out, attn_scores.squeeze(-1)


def train_model(model, X, y, epochs=500, lr=0.0001, batch_size=16, print_every=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    dataset_size = X.shape[0]
    
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        
        batch_pbar = tqdm(range(0, dataset_size, batch_size), 
                         desc=f"Epoch {epoch+1}/{epochs}", 
                         unit="batch",
                         leave=False)  
        
        for i in batch_pbar:
            indices = permutation[i:i+batch_size]
            X_batch = X[indices]
            y_batch = y[indices]
            
            optimizer.zero_grad()
            output, _ = model(X_batch)
            loss = criterion(output, y_batch)
            
            if torch.isnan(loss):
                print("NaN loss detected, skipping batch")
                continue
                
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
            
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})
        
        avg_loss = epoch_loss / dataset_size
        
        epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.8f}"})
    

    return model



def evaluate_model(model, X, y):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        output, attn_weights = model(X)
        loss = criterion(output, y)
    print(f"Final Loss: {loss.item():.8f}")
    print(f"Predicted: {output.squeeze().numpy()}")
    print(f"Target: {y.squeeze().numpy()}")
    return output, attn_weights, loss.item()


def sliding_window_importance(
    model, X_sample, y_target=None, window_size=5, stride=1
):
    """
    Calculates importance scores for each window in X_sample.
    Works even if sequence is short by using smaller windows.
    """
    model.eval()
    seq_len, num_features = X_sample.shape
    X_sample = X_sample.clone()

    with torch.no_grad():
        baseline_pred, _ = model(X_sample.unsqueeze(0))

    if y_target is not None:
        print(f"Baseline Prediction: {baseline_pred}\nTarget: {y_target}")

    importance_scores = []

    if seq_len < window_size:
        window_size = seq_len
        stride = 1

    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        X_masked = X_sample.clone()
        X_masked[start:end, :] = 0.0  
        with torch.no_grad():
            masked_pred, _ = model(X_masked.unsqueeze(0))
        score = torch.norm((baseline_pred - masked_pred) / (baseline_pred + 1e-8)).item()
        importance_scores.append((start, end, score))

    return importance_scores
