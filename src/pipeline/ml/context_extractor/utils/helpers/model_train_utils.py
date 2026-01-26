import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from typing import Optional, Sequence

from src.pipeline.ml.context_extractor.utils.models.attention_lstm import AttentionLSTM
from src.pipeline.ml.context_extractor.utils.models.attention_transformer import TransformerAttention
from src.pipeline.ml.context_extractor.utils.models.attention_mamba import AttentionMamba
from src.pipeline.ml.context_extractor.utils.models.attention_tcn import AttentionTCN
from src.pipeline.ml.context_extractor.utils.models.attention_tcn_lstm import AttentionTCNLSTM
from src.pipeline.ml.context_extractor.utils.models.attention_tcn_mamba import AttentionTCNMamba


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
        ).to(device)

    if model_type_norm in ("tcn_mamba", "tcn-mamba", "tcn+mamba"):
        return AttentionTCNMamba(
            input_features=input_features,
            n_predictions=n_predictions,
            output_features=output_features,
            hidden_dim=hidden_dim,
            tcn_layers=tcn_layers if tcn_layers is not None else lstm_layers,
            mamba_layers=mamba_layers if mamba_layers is not None else lstm_layers,
            d_state=mamba_d_state if mamba_d_state is not None else 16,
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
        ).to(device)

    if model_type_norm == "mamba":
        return AttentionMamba(
            input_features=input_features,
            n_predictions=n_predictions,
            output_features=output_features,
            hidden_dim=hidden_dim,
            mamba_layers=mamba_layers if mamba_layers is not None else lstm_layers,
            d_state=mamba_d_state if mamba_d_state is not None else 16,
            dropout=dropout,
            use_scalar=use_scalar,
            scalar_embedding_dim=scalar_embedding_dim,
            use_config=use_experiment_config,
            config_dim=config_dim,
            config_embedding_dim=config_embedding_dim,
        ).to(device)

    if model_type_norm == "transformer":
        return TransformerAttention(
            input_features=input_features,
            n_predictions=n_predictions,
            output_features=output_features
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
    ).to(device)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    feature_weights: Optional[torch.Tensor] = None,
    feature_loss_types: Optional[Sequence[str]] = None,
    extra_l2_reg: float = 0.0,
) -> float:
    model.train()
    train_loss = 0.0
    for Xb, Yb, sprinback, experiment_config in train_loader:
        # Move data to device
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        sprinback = sprinback.to(device)
        experiment_config = experiment_config.to(device)
        # Forward pass
        pred, _ = model(Xb, sprinback, experiment_config)
        if feature_weights is None and not feature_loss_types:
            loss = criterion(pred, Yb)
        else:
            if feature_loss_types:
                loss = 0.0
                weights = feature_weights.to(device) if feature_weights is not None else None
                for i, loss_type in enumerate(feature_loss_types):
                    if loss_type == "smoothl1":
                        part = F.smooth_l1_loss(pred[:, :, i], Yb[:, :, i])
                    else:
                        part = F.mse_loss(pred[:, :, i], Yb[:, :, i])
                    if weights is not None:
                        part = part * weights[i]
                    loss = loss + part
                loss = loss / len(feature_loss_types)
            else:
                weights = feature_weights.to(device).view(1, 1, -1)
                loss = ((pred - Yb) ** 2 * weights).mean()

        if extra_l2_reg > 0.0:
            l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + extra_l2_reg * l2_penalty
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate loss
        train_loss += loss.item()

    return train_loss / len(train_loader)


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    feature_weights: Optional[torch.Tensor] = None,
    feature_loss_types: Optional[Sequence[str]] = None,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    val_loss = 0.0
    val_preds_epoch = []
    val_targets_epoch = []

    with torch.no_grad():
        for Xb, Yb, springback, experiment_config in val_loader:
            Xb = Xb.to(device, non_blocking=True)
            Yb = Yb.to(device, non_blocking=True)
            springback = springback.to(device, non_blocking=True).view(-1, 1)
            experiment_config = experiment_config.to(device, non_blocking=True)

            pred, _ = model(Xb, springback, experiment_config)
            if feature_weights is None and not feature_loss_types:
                loss = criterion(pred, Yb)
            else:
                if feature_loss_types:
                    loss = 0.0
                    weights = feature_weights.to(device) if feature_weights is not None else None
                    for i, loss_type in enumerate(feature_loss_types):
                        if loss_type == "smoothl1":
                            part = F.smooth_l1_loss(pred[:, :, i], Yb[:, :, i])
                        else:
                            part = F.mse_loss(pred[:, :, i], Yb[:, :, i])
                        if weights is not None:
                            part = part * weights[i]
                        loss = loss + part
                    loss = loss / len(feature_loss_types)
                else:
                    weights = feature_weights.view(1, 1, -1)
                    loss = ((pred - Yb) ** 2 * weights).mean()

            val_loss += loss.item()

            val_preds_epoch.append(pred.detach().cpu())
            val_targets_epoch.append(Yb.detach().cpu())

    val_loss /= len(val_loader)
    return val_loss, torch.cat(val_preds_epoch, dim=0), torch.cat(val_targets_epoch, dim=0)


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
