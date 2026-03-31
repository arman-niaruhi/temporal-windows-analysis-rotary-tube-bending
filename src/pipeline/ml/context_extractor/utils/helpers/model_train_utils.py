import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from typing import Optional, Sequence

from src.pipeline.ml.context_extractor.utils.models.attention_lstm import AttentionLSTM
from src.pipeline.ml.context_extractor.utils.models.attention_tcn import AttentionTCN
from src.pipeline.ml.context_extractor.utils.models.attention_tcn_lstm import AttentionTCNLSTM



def create_model(
    input_features: int,
    n_predictions: int,
    output_features: int,
    hidden_dim: int,
    lstm_layers: int,
    dropout: float,
    device: torch.device,
    *,
    model_type: str = "lstm",
    use_scalar: bool = True,
    scalar_embedding_dim: int = 16,
    use_experiment_config: bool = True,
    config_dim: int | None = None,
    config_embedding_dim: int = 16,
    split_output_heads: bool = False,
    main_head_hidden_sizes: list[int] | None = None,
    secondary_head_hidden_sizes: list[int] | None = None,
    tcn_layers: int | None = None,
    tcn_kernel_size: int = 3,
    mamba_layers: int | None = None,
    mamba_d_state: int | None = None,
    use_attention: bool = True,
    use_feature_attention: bool = False,
    use_angle_embedding: bool = False,
    angle_embedding_dim: int = 8,
    attention_type: str = "mlp",
) -> nn.Module:
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

    model_type_norm = (model_type or "lstm").lower()

    if model_type_norm in ("tcn_lstm", "tcn-lstm", "tcn+lstm"):
        return AttentionTCNLSTM(
            input_features=input_features,
            n_predictions=n_predictions,
            output_features=output_features,
            hidden_dim=hidden_dim,
            tcn_layers=tcn_layers if tcn_layers is not None else lstm_layers,
            lstm_layers=lstm_layers,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
            use_scalar=use_scalar,
            scalar_embedding_dim=scalar_embedding_dim,
            use_config=use_experiment_config,
            config_dim=config_dim,
            config_embedding_dim=config_embedding_dim,
            split_output_heads=split_output_heads,
            main_head_hidden_sizes=main_head_hidden_sizes,
            secondary_head_hidden_sizes=secondary_head_hidden_sizes,
            use_attention=use_attention,
            use_feature_attention=use_feature_attention,
            use_angle_embedding=use_angle_embedding,
            angle_embedding_dim=angle_embedding_dim,
            attention_type=attention_type,
        ).to(device)


    if model_type_norm == "tcn":
        return AttentionTCN(
            input_features=input_features,
            n_predictions=n_predictions,
            output_features=output_features,
            hidden_dim=hidden_dim,
            tcn_layers=tcn_layers if tcn_layers is not None else lstm_layers,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
            use_scalar=use_scalar,
            scalar_embedding_dim=scalar_embedding_dim,
            use_config=use_experiment_config,
            config_dim=config_dim,
            config_embedding_dim=config_embedding_dim,
            split_output_heads=split_output_heads,
            main_head_hidden_sizes=main_head_hidden_sizes,
            secondary_head_hidden_sizes=secondary_head_hidden_sizes,
            use_angle_embedding=use_angle_embedding,
            angle_embedding_dim=angle_embedding_dim,
            attention_type=attention_type,
        ).to(device)

    return AttentionLSTM(
        input_features=input_features,
        n_predictions=n_predictions,
        output_features=output_features,
        hidden_dim=hidden_dim,
        lstm_layers=lstm_layers,
        dropout=dropout,
        use_scalar=use_scalar,
        scalar_embedding_dim=scalar_embedding_dim,
        use_config=use_experiment_config,
        config_dim=config_dim,
        config_embedding_dim=config_embedding_dim,
        split_output_heads=split_output_heads,
        main_head_hidden_sizes=main_head_hidden_sizes,
        secondary_head_hidden_sizes=secondary_head_hidden_sizes,
        use_feature_attention=use_feature_attention,
        use_angle_embedding=use_angle_embedding,
        angle_embedding_dim=angle_embedding_dim,
        attention_type=attention_type,
    ).to(device)

def compute_derivative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the MSE between the temporal derivatives of predictions and targets.
    Helps the model capture oscillations and sharp changes.
    """
    # pred/target shape: (batch, timesteps, features)
    if pred.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device)
    
    d_pred = pred[:, 1:, :] - pred[:, :-1, :]
    d_target = target[:, 1:, :] - target[:, :-1, :]
    return F.mse_loss(d_pred, d_target)

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    feature_weights: Optional[torch.Tensor] = None,
    feature_loss_types: Optional[Sequence[str]] = None,
    extra_l2_reg: float = 0.0,
    derivative_loss_weight: float = 0.0, # <--- NEW PARAMETER
) -> float:
    model.train()
    train_loss = 0.0
    for Xb, Yb, sprinback, experiment_config in train_loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        sprinback, experiment_config = sprinback.to(device), experiment_config.to(device)

        pred, _ = model(Xb, sprinback, experiment_config)
        
        # Standard Point-wise Loss
        if feature_loss_types:
            base_loss = 0.0
            weights = feature_weights.to(device) if feature_weights is not None else None
            for i, loss_type in enumerate(feature_loss_types):
                part = F.smooth_l1_loss(pred[:, :, i], Yb[:, :, i]) if loss_type == "smoothl1" else F.mse_loss(pred[:, :, i], Yb[:, :, i])
                if weights is not None:
                    part = part * weights[i]
                base_loss += part
            base_loss = base_loss / len(feature_loss_types)
        else:
            weights = feature_weights.to(device).view(1, 1, -1) if feature_weights is not None else 1.0
            base_loss = (criterion(pred, Yb) * weights).mean()

        # NEW: Temporal Derivative Loss
        loss = base_loss
        if derivative_loss_weight > 0.0:
            d_loss = compute_derivative_loss(pred, Yb)
            loss += derivative_loss_weight * d_loss

        if extra_l2_reg > 0.0:
            l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
            loss += extra_l2_reg * l2_penalty
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)

def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    feature_weights: Optional[torch.Tensor] = None,
    feature_loss_types: Optional[Sequence[str]] = None,
    derivative_loss_weight: float = 0.0, # <--- NEW PARAMETER
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    val_loss = 0.0
    val_preds_epoch, val_targets_epoch = [], []

    with torch.no_grad():
        for Xb, Yb, springback, experiment_config in val_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            springback = springback.view(-1, 1).to(device)
            experiment_config = experiment_config.to(device)

            pred, _ = model(Xb, springback, experiment_config)
            
            # Base Validation Loss
            if feature_loss_types:
                v_loss = 0.0
                weights = feature_weights.to(device) if feature_weights is not None else None
                for i, loss_type in enumerate(feature_loss_types):
                    part = F.smooth_l1_loss(pred[:, :, i], Yb[:, :, i]) if loss_type == "smoothl1" else F.mse_loss(pred[:, :, i], Yb[:, :, i])
                    if weights is not None:
                        part = part * weights[i]
                    v_loss += part
                v_loss /= len(feature_loss_types)
            else:
                v_loss = criterion(pred, Yb).item()

            # Add Derivative Component to Val Loss
            if derivative_loss_weight > 0.0:
                v_loss += derivative_loss_weight * compute_derivative_loss(pred, Yb).item()

            val_loss += v_loss
            val_preds_epoch.append(pred.cpu())
            val_targets_epoch.append(Yb.cpu())

    return val_loss / len(val_loader), torch.cat(val_preds_epoch, dim=0), torch.cat(val_targets_epoch, dim=0)


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
        "Patience": f"{patience}/20",
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
        for Xb, Yb, springback, experiment_config in val_loader:
            Xb = Xb.to(device)
            springback = springback.to(device).view(-1, 1)
            experiment_config = experiment_config.to(device)
            pred, _ = model(Xb, springback, experiment_config)
            all_preds.append(pred.cpu())
            all_targets.append(Yb)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_targets, all_preds
