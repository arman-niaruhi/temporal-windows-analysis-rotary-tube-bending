import gc

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from src.pipeline.ml.context_extractor.utils.models.attention_lstm import AttentionLSTM


def create_model(input_features: int, n_predictions: int, output_features: int,
                 hidden_dim: int, lstm_layers: int, dropout: float,
                 device: torch.device) -> AttentionLSTM:
    """
    Instantiate and initialize the Attention LSTM model.
    
    Args:
        input_features: Number of input features per timestep
        n_predictions: Number of timesteps to predict
        output_features: Number of output features per prediction
        hidden_dim: Hidden dimension size of the LSTM layers
        lstm_layers: Number of stacked LSTM layers
        dropout: Dropout probability
        device: torch device (CPU or CUDA)

    Returns:
        AttentionLSTM model moved to the specified device
    """
    return AttentionLSTM(
        input_features=input_features,
        n_predictions=n_predictions,
        output_features=output_features,
        hidden_dim=hidden_dim,
        lstm_layers=lstm_layers,
        dropout=dropout,
    ).to(device)



def train_one_epoch(model: nn.Module, train_loader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: nn.Module,
                    device: torch.device) -> float:
    """
    Perform a single epoch of training.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: torch device

    Returns:
        Average training loss for the epoch
    """
    model.train()  # set model to training mode
    train_loss = 0.0

    for Xb, Yb in train_loader:
        # Move data to device
        Xb, Yb = Xb.to(device), Yb.to(device)
        
        # Forward pass
        pred, _ = model(Xb)
        loss = criterion(pred, Yb)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        # Accumulate loss
        train_loss += loss.item()
        
        # Free memory
        del pred, loss, Xb, Yb

    # Garbage collection and clear CUDA cache to prevent memory leaks
    gc.collect()
    torch.cuda.empty_cache()

    return train_loss / len(train_loader)



def validate_one_epoch(model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module, device: torch.device) -> tuple:
    """
    Evaluate model on validation data for one epoch.

    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: torch device

    Returns:
        tuple: (average validation loss, all predictions, all targets)
    """
    model.eval()  # set model to evaluation mode
    val_loss = 0.0
    val_preds_epoch = []
    val_targets_epoch = []

    with torch.no_grad():  # no gradients for validation
        for Xb, Yb in val_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            pred, _ = model(Xb)
            val_loss += criterion(pred, Yb).item()
            
            val_preds_epoch.append(pred.cpu())
            val_targets_epoch.append(Yb.cpu())

    val_loss /= len(val_loader)
    val_preds_epoch = torch.cat(val_preds_epoch, dim=0)
    val_targets_epoch = torch.cat(val_targets_epoch, dim=0)
    
    return val_loss, val_preds_epoch, val_targets_epoch



def format_progress_bar(train_loss: float, val_loss: float, metrics: dict,
                        best_val_loss: float, current_lr: float, patience: int) -> dict:
    """
    Prepare a dictionary of metrics for tqdm progress bar display.

    Args:
        train_loss: Training loss for current epoch
        val_loss: Validation loss for current epoch
        metrics: Dictionary of validation metrics
        best_val_loss: Best validation loss so far
        current_lr: Current learning rate
        patience: Early stopping patience counter

    Returns:
        Dictionary with formatted metrics for display
    """
    return {
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
    }



def evaluate_final_model(model: nn.Module, val_loader: DataLoader, 
                         device: torch.device) -> tuple:
    """
    Evaluate the model on the entire validation dataset.

    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        device: torch device

    Returns:
        tuple: (all validation targets, all validation predictions)
    """
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

    return all_targets, all_preds
