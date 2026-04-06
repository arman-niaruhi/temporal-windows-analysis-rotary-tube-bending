# ============================
# Standard library imports
# ============================
import time
import warnings
from tqdm import tqdm
from pathlib import Path      
import logging
from typing import Any
import numpy as np
import json

# ============================
# Third-party ML utilities
# ============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# ============================
# Model interpretability modules
# ============================
from src.pipeline.ml.context_extractor.utils.algorithms.feature_importance import (
    analyze_feature_importance
)
from src.pipeline.ml.context_extractor.utils.algorithms.inetgrated_gradients import (
    save_integrated_gradients_combined
)
from src.pipeline.ml.context_extractor.utils.algorithms.window_based_algorithm import (
    save_window_importance_results,
    window_based_importance
)
from src.pipeline.ml.context_extractor.utils.algorithms.timestep_sensitivity import (
    run_all_timestep_sensitivity,
)

# ============================
# Data loading utilities
# ============================
from src.pipeline.ml.context_extractor.utils.data.data_preprocessor import (
    create_data_loaders
)
from src.pipeline.ml.context_extractor.utils.helpers.seed_utils import (
    enforce_reproducibility,
)

# ============================
# Visualization utilities
# ============================
from src.pipeline.ml.context_extractor.utils.plots.plot_utils import (
    get_plot_batch,
    compute_plot_limits
)
from src.pipeline.ml.context_extractor.utils.plots.plot_metrics import (
    plot_all_metrics
)
from src.pipeline.ml.context_extractor.utils.plots.plot_window_impotance import (
    visualize_window_importance,
    visualize_window_importance_heatmap,
    visualize_window_importance_with_sensors,
) 
from src.pipeline.ml.context_extractor.utils.plots.plot_attention import (
    generate_final_attention_plot,
)
from src.pipeline.ml.context_extractor.utils.plots.plot_epoch_results import (
    generate_epoch_plots,
    save_validation_scatter
)

# ============================
# Training and evaluation helpers
# ============================
from src.pipeline.ml.context_extractor.utils.helpers.model_train_utils import (
    validate_one_epoch,
    format_progress_bar,
    evaluate_final_model,
    train_one_epoch,
    create_model
)

# ============================
# Metrics computation
# ============================
from src.pipeline.ml.context_extractor.utils.helpers.metrics_utils import (
    compute_all_metrics,
    compute_epoch_metrics,
)

# Initialize module-level logger
logger = logging.getLogger(__name__)

RESULTS_COPY_DIR = Path("results") / "temporal_pattern_analysis"


def _build_experiment_description(
    process_part: str,
    params: dict,
    preprocessing_info: dict,
    X: torch.Tensor,
    Y: torch.Tensor,
    target_feature_names: list[str],
    input_path_param: dict | None = None,
    general_setting: dict | None = None,
) -> tuple[str, int, int, int]:
    n_experiments, timesteps_in, features_in = X.shape
    _, n_crosscut, features_out = Y.shape
    split_config_path = preprocessing_info.get(
        "split_config_path",
        "config/data-split-config/train_test_split.json",
    )
    model_type = params.get("model_type", "unknown")
    experiment_description = f"""
    SPLIT CONFIG: {split_config_path}
    PART: {process_part}
    MODEL TYPE: {model_type}
    ================== GENERAL SETTINGS ==================
    {general_setting}
    ==================== INPUT PATHS =====================
    {input_path_param}
    ============== PREPROCESSING INFO ====================
    {preprocessing_info}
    ==================== MODEL INFO ======================
    INPUT OF TRAIN:
    N_EXPERIMENTS, N_TIMESTEPS_IN, N_FEATURES_IN = ({n_experiments}, {timesteps_in}, {features_in})
    OUTPUT_VALIDATION:
    N_EXPERIMENTS, N_CROSSCUT, FEATURES_OUT = ({n_experiments}, {n_crosscut}, {features_out})

    GEOMETRY_FEATURES: {target_feature_names}
    ================== TRAINING INFO =====================
    {params}
    """
    return experiment_description, features_in, n_crosscut, features_out


def _save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_python_types(payload), f, indent=2, ensure_ascii=False)


def _to_python_types(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_python_types(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python_types(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return _to_python_types(value.detach().cpu().numpy())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _save_experiment_description_local(description: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_description.txt").write_text(description, encoding="utf-8")


def _save_model_local(model: nn.Module, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output_dir / "model_state_dict.pth")


def _save_final_metrics_local(
    all_targets: torch.Tensor,
    all_preds: torch.Tensor,
    val_losses: list[float],
    epoch_times: list[float],
    output_dir: Path,
) -> None:
    final_metrics = compute_all_metrics(all_targets, all_preds)
    payload = {
        **final_metrics,
        "total_epochs": len(val_losses),
        "avg_epoch_time": float(np.mean(epoch_times)) if epoch_times else 0.0,
    }
    _save_json(payload, output_dir / "final_metrics.json")


def _save_feature_importance_local(combined_importance_df: dict, output_dir: Path) -> None:
    if combined_importance_df is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = {
        key: value if isinstance(value, list) else [value]
        for key, value in combined_importance_df.items()
    }
    import pandas as pd

    pd.DataFrame(normalized).to_csv(
        output_dir / "feature_importance_summary.csv",
        index=False,
    )


def update_best_model(
    val_loss: float,
    best_val_loss: float,
    model: nn.Module,
    patience: int,
    epoch: int,
) -> tuple[float, dict[str, torch.Tensor] | None, int]:
    """Track the best validation checkpoint locally."""
    del epoch
    if val_loss < best_val_loss - 1e-6:
        return val_loss, model.state_dict(), 0
    return best_val_loss, None, patience + 1

def _model_uses_attention(model: nn.Module) -> bool:
    return bool(getattr(model, "use_attention", True))

def _capture_tcn_activations(
    model: nn.Module,
    x: torch.Tensor,
    springback: torch.Tensor,
    experiment_config: torch.Tensor,
) -> torch.Tensor | None:
    if not hasattr(model, "tcn"):
        return None
    activations: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        activations["tcn"] = output

    handle = model.tcn.register_forward_hook(_hook)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            _ = model(x, springback, experiment_config)
    finally:
        handle.remove()
        if was_training:
            model.train()
    return activations.get("tcn")

def _maybe_denormalize_targets(
    targets: torch.Tensor,
    target_scaler,
) -> torch.Tensor:
    if target_scaler is None:
        return targets
    shape = targets.shape
    flat = targets.detach().cpu().numpy().reshape(-1, shape[-1])
    denorm = target_scaler.inverse_transform(flat).reshape(shape)
    return torch.from_numpy(denorm).float()

def _log_target_stats(
    label: str,
    targets: torch.Tensor,
    target_scaler,
    target_feature_names: list[str],
) -> None:
    stats_source = _maybe_denormalize_targets(targets, target_scaler)
    per_feature_std = stats_source.std(dim=(0, 1), unbiased=False).cpu().numpy().tolist()
    per_feature_mean = stats_source.mean(dim=(0, 1)).cpu().numpy().tolist()
    names = target_feature_names or [f"feature_{i}" for i in range(len(per_feature_std))]
    parts = []
    for i, name in enumerate(names):
        if i >= len(per_feature_std):
            break
        parts.append(f"{name}: mean={per_feature_mean[i]:.4f}, std={per_feature_std[i]:.4f}")
    logger.info(f"{label} target stats: " + " | ".join(parts))


def train_model(
    X_train: torch.Tensor, 
    Y_train: torch.Tensor, 
    X_test: torch.Tensor, 
    Y_test: torch.Tensor,
    springbacks_train: torch.Tensor,
    springbacks_test: torch.Tensor,
    experiment_configurations_train: torch.Tensor,
    experiment_configurations_test: torch.Tensor,
    params: dict,
    occlusion_params: dict | None,
    sensor_names: list[str],
    target_feature_names: list[str],
    process_part: str,
    preprocessing_info: dict,
    annot_timesteps: list[int],
    mandrel_extraction_annot_timesteps: list[int],
    target_scaler = None,
    input_path_param: dict | None = None,
    general_setting: dict | None = None,
) -> Any:
    """
    Main training and analysis routine for the Attention-LSTM model.

    The function trains a new model and stores all generated artifacts locally.
    """

    occlusion_analysis_enabled = occlusion_params is not None
    occlusion_params = occlusion_params or {}
    general_setting = general_setting or {}
    random_seed = general_setting.get("seed", 42)
    enforce_reproducibility(seed=random_seed)

    window_size = str(preprocessing_info.get("window_num", 0))
    split_config_path = preprocessing_info.get(
        "split_config_path",
        "config/data-split-config/train_test_split.json",
    )
    model_type = params.get("model_type", "unknown")
    dataset_type = Path(split_config_path).stem if split_config_path else "default"
    parts = [process_part, model_type, dataset_type]
    if preprocessing_info.get("resample", False):
        parts.append("resample")
        parts.append(f"ws{window_size}")
    run_name = "_".join(parts)
    
    # ============================================================
    # Directory structure for storing generated artifacts
    # ============================================================
    base_dir = Path(params.get("results_dir", RESULTS_COPY_DIR)) / run_name
    process_part = process_part
    
    predictions_dir = base_dir / "01_predictions"
    attention_dir = base_dir / "02_attention"
    attention_csv_dir = base_dir / "02_attention_csv"
    attention_lines_dir = base_dir / "03_attention_lines"
    feature_importance_dir = base_dir / "04_feature_importance"
    integrated_gradients_dir= base_dir/ "05_integrated_gradients" 
    window_importance_plots_dir = base_dir / "06_window_importance"
    training_metric_results_dir = base_dir / "07_metrics"
    val_pred_dir = base_dir / "08_val_predictions"


    # Create directories if they do not exist
    for d in [
        attention_dir,
        predictions_dir, 
        attention_csv_dir, 
        attention_lines_dir,
        feature_importance_dir, 
        integrated_gradients_dir,
        window_importance_plots_dir,
        training_metric_results_dir,
        val_pred_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    
    
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================================================
    # Experiment setup
    # ============================================================
    experiment_desc, features_in, predictions_out, features_out = \
        _build_experiment_description(
            process_part,
            params,
            preprocessing_info,
            X_train,
            Y_train,
            target_feature_names,
            input_path_param=input_path_param,
            general_setting=general_setting,
        )

    _log_target_stats("Train", Y_train, target_scaler, target_feature_names)
    _log_target_stats("Val", Y_test, target_scaler, target_feature_names)
    
    # Create PyTorch DataLoaders using customized dataloader
    train_loader, val_loader, plot_loader = create_data_loaders(
        X_train, Y_train, X_test, Y_test, springbacks_train, springbacks_test, 
        experiment_configurations_train, experiment_configurations_test, params["batch_size"],
        random_seed=random_seed,
    )

    # Extract a fixed batch for visualization
    plot_X, plot_Y, springback, experiment_config = get_plot_batch(plot_loader, device)
    plot_train_X = None
    if len(train_loader.dataset) > 0:
        plot_train_loader = DataLoader(
            train_loader.dataset,
            batch_size=min(64, len(train_loader.dataset)),
            shuffle=False,
        )
        plot_train_X = get_plot_batch(
            plot_train_loader,
            device,
        )
    # Fall back to validation batch if train batch is unavailable
    if plot_train_X is None:
        plot_train_X = plot_X
    n_samples = min(4, len(plot_Y))
    
    
    # Compute y-axis limits for consistent plotting
    y_lim = compute_plot_limits(Y_test)

    _save_experiment_description_local(experiment_desc, base_dir)
    _save_json(
        {
            "generalSetting": general_setting,
            "inputPathParams": input_path_param,
            "preprocessingParams": preprocessing_info,
            "trainingParams": params,
            "dataset_name": dataset_type,
            "train_size": len(X_train),
            "val_size": len(X_test),
        },
        base_dir / "run_config.json",
    )
        
    y_lim_source = _maybe_denormalize_targets(Y_test, target_scaler)
    y_lim = compute_plot_limits(y_lim_source)
    plot_X, plot_Y, springback, experiment_config = get_plot_batch(plot_loader, device)
    n_samples = min(4, len(plot_Y))
        
    # Initialize model
    use_experiment_config = bool(params.get("use_experiment_config", False))
    use_scalar = bool(params.get("use_scalar", False))
    use_attention = bool(params.get("use_attention", False))
    config_dim = experiment_configurations_train.shape[1] if use_experiment_config else None
    model = create_model(
        features_in,
        predictions_out,
        features_out,
        params["hidden_dim"],
        params["lstm_layers"],
        params["dropout"],
        device,
        model_type=params.get("model_type", "lstm"),
        use_experiment_config=use_experiment_config,
        config_dim=config_dim,
        use_scalar=use_scalar,
        split_output_heads=bool(params.get("split_output_heads", False)),
        main_head_hidden_sizes=params.get("main_head_hidden_sizes"),
        secondary_head_hidden_sizes=params.get("secondary_head_hidden_sizes"),
        tcn_layers=params.get("tcn_layers"),
        tcn_kernel_size=params.get("tcn_kernel_size", 3),
        use_attention=use_attention,
        use_feature_attention=bool(params.get("use_feature_attention", False)),
        use_angle_embedding=bool(params.get("use_angle_embedding", False)),
        angle_embedding_dim=int(params.get("angle_embedding_dim", 8)),
        attention_type=params.get("attention_type", "mlp"),
    )
    saved_model_path = base_dir / "model" / "model_state_dict.pth"
        
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5, threshold=1e-4)
    # Loss function
    criterion = nn.MSELoss()

    feature_weights = None
    feature_weights_base = None
    if params.get("feature_loss_weights"):
        feature_weights_base = torch.tensor(
            params["feature_loss_weights"], dtype=torch.float32, device=device
        )
        feature_weights_base = feature_weights_base / feature_weights_base.mean()
    elif params.get("auto_feature_loss_weighting"):
        per_feature_std = Y_train.std(dim=(0, 1)).to(device)
        feature_weights_base = 1.0 / (per_feature_std + 1e-8)
        feature_weights_base = feature_weights_base / feature_weights_base.mean()

    dynamic_feature_loss_weighting = bool(params.get("dynamic_feature_loss_weighting", False))
    dynamic_warmup_epochs = int(params.get("dynamic_feature_loss_warmup_epochs", 0))
    dynamic_end_weights = params.get("dynamic_feature_loss_end_weights")
    if dynamic_feature_loss_weighting and dynamic_warmup_epochs <= 0:
        dynamic_feature_loss_weighting = False

    feature_loss_types = None
    if params.get("feature_loss_types"):
        feature_loss_types = list(params["feature_loss_types"])
        if len(feature_loss_types) == 1 and features_out > 1:
            feature_loss_types = feature_loss_types * features_out
        if len(feature_loss_types) != features_out:
            raise ValueError(
                "feature_loss_types length must match output features."
            )
    elif params.get("use_smoothl1_for_secondary"):
        feature_loss_types = ["smoothl1" if i == 0 else "mse" for i in range(features_out)]

    extra_l2_reg = float(params.get("extra_l2_reg", 0.0))
        
        
    # Training state tracking
    val_losses, train_losses, learning_rates, epoch_times = [], [], [], []
    metrics_history = {
        'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'mape': [],
        'max_error': [], 'evs': [], 'mbe': [], 'medae': []
    }
    best_val_loss = float("inf")
    best_state = None
    patience = 0
        
        
    last_epoch = None
    if not params.get("train"):
        if not saved_model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found for train=False: {saved_model_path}"
            )
        state_dict = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Loaded existing local model from %s", saved_model_path)
    else:
        # Training loop
        fpbar = tqdm(range(1, params["max_epochs"] + 1), desc="Training")
        for epoch in fpbar:
            epoch_start = time.time()

            feature_weights = feature_weights_base
            if dynamic_feature_loss_weighting and features_out > 1:
                start_weights = (
                    feature_weights_base
                    if feature_weights_base is not None
                    else torch.ones(features_out, device=device)
                )
                if dynamic_end_weights is not None:
                    if len(dynamic_end_weights) != features_out:
                        raise ValueError("dynamic_feature_loss_end_weights length must match output features.")
                    end_weights = torch.tensor(dynamic_end_weights, dtype=torch.float32, device=device)
                else:
                    end_weights = torch.ones_like(start_weights)
                t = min(1.0, epoch / float(dynamic_warmup_epochs))
                feature_weights = (1.0 - t) * start_weights + t * end_weights
                feature_weights = feature_weights / feature_weights.mean()
            
            # Train and validate
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                feature_weights,
                feature_loss_types,
                extra_l2_reg,
            )
            val_loss, val_preds, val_targets = validate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                feature_weights,
                feature_loss_types,
            )

            val_preds_denorm = _maybe_denormalize_targets(val_preds, target_scaler)
            val_targets_denorm = _maybe_denormalize_targets(val_targets, target_scaler)

            save_validation_scatter(
                val_targets=val_targets_denorm,
                val_preds=val_preds_denorm,
                target_feature_names=target_feature_names,
                val_pred_dir=val_pred_dir,
                epoch=epoch,
            )
            
            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            
            # Compute evaluation metrics
            metrics = compute_epoch_metrics(val_targets_denorm, val_preds_denorm)
            for key in metrics_history.keys():
                metrics_history[key].append(metrics[key])
            
            per_feature_mse = metrics.get("per_feature_mse") or []
            per_feature_r2 = metrics.get("per_feature_r2") or []
            if per_feature_mse and per_feature_r2:
                feature_labels = target_feature_names or [f"feature_{i}" for i in range(len(per_feature_mse))]
                feature_metrics_parts = []
                for i, name in enumerate(feature_labels):
                    if i >= len(per_feature_mse) or i >= len(per_feature_r2):
                        break
                    feature_metrics_parts.append(
                        f"{name}: MSE={per_feature_mse[i]:.6f}, R2={per_feature_r2[i]:.4f}"
                    )
                tqdm.write(f"Epoch {epoch} per-feature metrics: " + " | ".join(feature_metrics_parts))
            
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            learning_rates.append(current_lr)
            
            
            # Epoch timing
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            
            # Track best model state
            best_val_loss, new_state, patience = update_best_model(
                val_loss, best_val_loss, model, patience, epoch
            )
            if new_state is not None:
                best_state = new_state
                
            last_epoch = epoch
            
            progress_info = format_progress_bar(
                train_loss, val_loss, metrics, best_val_loss, current_lr, patience
            )
            
            # Update progress bar
            fpbar.set_postfix(progress_info, refresh=True)
            
            
            # Early stopping condition
            if patience >= 15:
                break
        
        
    if last_epoch is not None:
        generate_epoch_plots(
                model=model,
                plot_X=plot_X,
                plot_Y=plot_Y,
                springback=springback,
                experiment_config=experiment_config,
                X_train=X_train,
                sensor_names=sensor_names,
                target_feature_names=target_feature_names,
                val_losses=val_losses,
                process_part=process_part,
                train_losses=train_losses,
                epoch=last_epoch,
                y_lim=y_lim,
                predictions_out=predictions_out,
                predictions_dir=predictions_dir,
                attention_csv_dir=attention_csv_dir,
                attention_dir=attention_dir,
                annot_timesteps=annot_timesteps,
                mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
                n_samples=n_samples,
                target_scaler=target_scaler,
            )

    # Load best-performing model
    if best_state is not None:
        model.load_state_dict(best_state)
        
        
    # Final evaluation
    all_targets, all_preds = evaluate_final_model(model, val_loader, device)
    all_targets = _maybe_denormalize_targets(all_targets, target_scaler)
    all_preds = _maybe_denormalize_targets(all_preds, target_scaler)
    _save_final_metrics_local(
        all_targets,
        all_preds,
        val_losses,
        epoch_times,
        base_dir,
    )

        
    # Persist trained model
    if params.get("train"):
        _save_model_local(model, base_dir / "model")
        logger.info("Model training completed and stored locally in %s.", base_dir)
    else:
        logger.info("Evaluation completed using existing local model in %s.", base_dir)
        
        
    # Generate final metrics plots
    split_config_path = preprocessing_info.get(
        "split_config_path",
        "config/data-split-config/train_test_split.json",
    )
    if train_losses and val_losses:
        plot_all_metrics(
            metrics_history,
            train_losses,
            val_losses,
            learning_rates,
            epoch_times,
            training_metric_results_dir,
            split_config_path=split_config_path,
        )
        
        
    # Feature importance and interpretability
    combined_importance_df = None
    if use_attention:
        combined_importance_df, _, _ = analyze_feature_importance(
            model=model,
            val_loader=val_loader,
            feature_names=sensor_names,
            saving_dir=feature_importance_dir,
        )
        
    if combined_importance_df is not None:
        X_sample = plot_X[:1]
        sensor_data_sample = X_test[:1].cpu().numpy()
        save_integrated_gradients_combined(
            model,
            X_sample,
            springback[:1],
            experiment_config[:1],
            sensor_data_sample,
            sensor_names,
            target_feature_names,
            integrated_gradients_dir,
            process_part,
            annot_timesteps,
            mandrel_extraction_annot_timesteps,
        )
        _save_feature_importance_local(
            combined_importance_df,
            feature_importance_dir,
        )
        
    if use_attention:
        generate_final_attention_plot(
                model=model,
                plot_X=plot_X,
                springback=springback,
                experiment_config=experiment_config,
                X_val=X_test,
                sensor_names=sensor_names,
                machine_part=process_part,
                attention_lines_dir=attention_lines_dir,
                annot_timesteps=annot_timesteps,
                mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
                target_feature_names=target_feature_names,
            )
        
    # Window-based occlusion importance (all angles)
    run_window_importance = (
        occlusion_analysis_enabled
        and occlusion_params.get("window_importance_enabled", True)
    )
    if run_window_importance:
        occluded_window_size = occlusion_params.get("occlusion_window_size", 10)
        stride = occlusion_params.get("occlusion_stride", 5)
        window_importance_output_dir = (
            window_importance_plots_dir / f"size{occluded_window_size}-stride{stride}"
        )
        n_angles = int(Y_test.shape[1])
        all_importance_data = []
        all_mean_importances = []

        for n_angle in range(n_angles):
            importance_df, mean_importance = window_based_importance(
                model=model,
                train_loader=val_loader,
                n_angle=n_angle,
                occluded_window_size=occluded_window_size,
                stride=stride,
                device=device,
            )

            visualize_window_importance(
                angle=n_angle,
                feature_names=sensor_names,
                mean_importance=mean_importance,
                annot_timesteps=annot_timesteps,
                window_importance_plots_dir=window_importance_plots_dir,
                mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
                process_part=process_part,
                occluded_window_size=occluded_window_size,
                stride=stride,
            )
            all_importance_data.append((n_angle, importance_df, mean_importance))
            all_mean_importances.append(mean_importance)

        if all_mean_importances:
            mean_importance_matrix = np.vstack(all_mean_importances)
            visualize_window_importance_heatmap(
                mean_importance_matrix=mean_importance_matrix,
                window_importance_plots_dir=window_importance_plots_dir,
                annot_timesteps=annot_timesteps,
                mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
                process_part=process_part,
                occluded_window_size=occluded_window_size,
                stride=stride,
            )
            visualize_window_importance_with_sensors(
                sensor_data=X_test.detach().cpu().numpy(),
                sensor_names=sensor_names,
                mean_importance_matrix=mean_importance_matrix,
                window_importance_plots_dir=window_importance_plots_dir,
                annot_timesteps=annot_timesteps,
                mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
                process_part=process_part,
                occluded_window_size=occluded_window_size,
                stride=stride,
            )

        save_window_importance_results(all_importance_data, window_importance_output_dir)
