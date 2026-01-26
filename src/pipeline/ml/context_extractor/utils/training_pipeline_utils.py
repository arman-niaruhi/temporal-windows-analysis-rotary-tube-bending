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

# ============================
# Third-party ML utilities
# ============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================
# Experiment tracking
# ============================
import mlflow
import mlflow.pytorch

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
# MLflow helper utilities
# ============================
from src.pipeline.ml.context_extractor.utils.helpers.mlflow_utils import (
    setup_mlflow_experiment,
    log_epoch_metrics,
    log_feature_importance_to_mlflow,
    log_model_parameters,
    find_previous_mlflow_run,
    log_final_metrics,
    move_images_to_mlflow_artifacts,
    save_experiment_description_as_text,
    update_best_model
)

# ============================
# Data loading utilities
# ============================
from src.pipeline.ml.context_extractor.utils.data.data_preprocessor import (
    create_data_loaders
)

# ============================
# Visualization utilities
# ============================
from src.pipeline.ml.context_extractor.utils.plots.plot_attention import (
    generate_final_attention_plot,
)
from src.pipeline.ml.context_extractor.utils.plots.plot_utils import (
    get_plot_batch,
    compute_plot_limits
)
from src.pipeline.ml.context_extractor.utils.plots.plot_metrics import (
    plot_all_metrics
)
from src.pipeline.ml.context_extractor.utils.plots.plot_window_impotance import (
    visualize_window_importance
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
    compute_epoch_metrics
)

# Initialize module-level logger
logger = logging.getLogger(__name__)

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
    occlusion_params: dict,
    sensor_names: list[str],
    target_feature_names: list[str],
    process_part: str,
    preprocessing_info: dict,
    annot_timesteps: list[int],
    mandrel_extraction_annot_timesteps: list[int],
    target_scaler = None,
) -> Any:
    """
    Main training and analysis routine for the Attention-LSTM model.

    The function supports two modes:
    1) Training a new model from scratch
    2) Resuming an existing MLflow run for analysis and visualization only
    """
    
    # ============================================================
    # Directory structure for storing generated artifacts
    # ============================================================
    base_dir = Path("images")
    process_part = process_part
    
    predictions_dir = base_dir / "01_predictions"
    loss_dir = base_dir / "02_loss"
    attention_dir = base_dir / "03_attention"
    attention_csv_dir = base_dir / "03_attention_csv"
    attention_lines_dir = base_dir / "04_attention_lines"
    feature_importance_dir = base_dir / "05_feature_importance"
    integrated_gradients_dir= base_dir/ "06_integrated_gradients" 
    window_importance_plots_dir = base_dir / "07_window_importance"
    training_metric_results_dir = base_dir / "08_metrics"
    diff_hist_dir = base_dir / "09_diff_hist"
    diff_bar_dir = base_dir / "10_diff_bar"
    val_pred_dir = base_dir / "11_val_predictions"
    val_diff_bar_dir = base_dir / "12_val_diff_bar"
    timestep_sensitivity_dir = base_dir / "13_timestep_sensitivity"


    # Create directories if they do not exist
    for d in [
        loss_dir, 
        attention_dir,
        predictions_dir, 
        attention_csv_dir, 
        attention_lines_dir,
        feature_importance_dir, 
        integrated_gradients_dir,
        window_importance_plots_dir,
        training_metric_results_dir,
        diff_hist_dir,
        diff_bar_dir,
        val_pred_dir,
        val_diff_bar_dir,
        timestep_sensitivity_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    
    
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    excluded58 = "" if not preprocessing_info["to_58_excluded"] else "58"
    window_size = str(preprocessing_info["window_num"])

    # ============================================================
    # MLflow experiment setup
    # ============================================================
    experiment_desc, features_in, predictions_out, features_out = \
        setup_mlflow_experiment(
            process_part, params, preprocessing_info, X_train, Y_train, target_feature_names
        )  
    
    # Create PyTorch DataLoaders using customized dataloader
    train_loader, val_loader, plot_loader = create_data_loaders(
        X_train, Y_train, X_test, Y_test, springbacks_train, springbacks_test, 
        experiment_configurations_train, experiment_configurations_test, params["batch_size"]
    )

    # Extract a fixed batch for visualization
    plot_X, plot_Y, springback, experiment_config = get_plot_batch(plot_loader, device)
    n_samples = min(4, len(plot_Y))
    
    
    # Compute y-axis limits for consistent plotting
    y_lim = compute_plot_limits(Y_test)

    # ============================================================
    # CASE 1 — TRAIN = FALSE → RESUME EXISTING RUN
    # ============================================================
    if not params.get("train"):
        run_id, model_uri = find_previous_mlflow_run(
            process_part, preprocessing_info
        )

        if run_id is None:
            logger.warning("No matching MLflow run found to resume.")
            return

        logger.info(f"Resuming MLflow run_id={run_id}")

        with mlflow.start_run(run_id=run_id):
            # Load previously trained model
            model = mlflow.pytorch.load_model(model_uri, map_location=device)


            # Generate final attention visualization using the loaded model
            generate_final_attention_plot(
                model,
                plot_X,
                springback,
                experiment_config,
                X_test,
                sensor_names,
                process_part,
                attention_lines_dir,
                annot_timesteps,
                mandrel_extraction_annot_timesteps,
            )


            # Compute integrated gradients if importance is available using the loaded model
            combined_importance_df, _, importance_paths = analyze_feature_importance(
                model=model,
                val_loader=val_loader,
                feature_names=sensor_names,
                saving_dir=feature_importance_dir
            )

            if combined_importance_df is not None:
                X_sample = plot_X[-1]
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
                    
                log_feature_importance_to_mlflow(
                    combined_importance_df
                )
               
               
            if params.get("timestep_sensitivity", False):
                run_all_timestep_sensitivity(
                    model=model,
                    val_loader=val_loader,
                    X_sample=plot_X[-1:],
                    springback_sample=springback[-1:],
                    experiment_config=experiment_config[-1:],
                    device=device,
                    saving_dir=timestep_sensitivity_dir,
                    occluded_window_size=occlusion_params.get("occlusion_window_size", 10),
                    stride=occlusion_params.get("occlusion_stride", 5),
                    annot_timesteps=annot_timesteps,
                    mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
                    sensor_data=plot_X[-1].detach().cpu().numpy(),
                    sensor_names=sensor_names,
                )

            # Window-based occlusion importance analysis using the loaded model and plot them
            all_importance_data = []
            for n_angle in range(46):
                importance_df, mean_importance = \
                    window_based_importance(
                        model=model,
                        train_loader=val_loader,
                        n_angle = n_angle,
                        occluded_window_size=occlusion_params.get("occlusion_window_size", 10),
                        stride=occlusion_params.get("occlusion_stride", 5),
                        device=device
                    )
                
                visualize_window_importance(
                angle=n_angle,
                feature_names=sensor_names,
                mean_importance=mean_importance,
                annot_timesteps=annot_timesteps,
                window_importance_plots_dir=window_importance_plots_dir,
                mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
                process_part=process_part,
                occluded_window_size=occlusion_params.get("occlusion_window_size", 10),
                stride=occlusion_params.get("occlusion_stride", 5)
            )

            all_importance_data.append((n_angle, importance_df, mean_importance))

            # Persist occlusion results for all angles
            save_window_importance_results(all_importance_data, window_importance_plots_dir)
 
            move_images_to_mlflow_artifacts(base_dir)

        logger.info("Training skipped; plots replaced in original MLflow run.")
        return

    # ============================================================
    # CASE 2 — TRAIN = TRUE → NEW RUN and Train a new model
    # ============================================================
    with mlflow.start_run(run_name=f"{process_part}_{excluded58}_ws{window_size}"):
        # Log experiment metadata
        save_experiment_description_as_text(experiment_desc)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_test))
        
        y_lim_source = _maybe_denormalize_targets(Y_test, target_scaler)
        y_lim = compute_plot_limits(y_lim_source)
        plot_X, plot_Y, springback, experiment_config = get_plot_batch(plot_loader, device)
        n_samples = min(4, len(plot_Y))
        
        # Initialize model
        use_experiment_config = bool(params.get("use_experiment_config", True))
        use_scalar = bool(params.get("use_scalar", True))
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
            mamba_layers=params.get("mamba_layers"),
            mamba_d_state=params.get("mamba_d_state"),
        )
        
        # Log model architecture parameters
        log_model_parameters(model)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=7, factor=0.5, threshold=1e-4)
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
            
            
            # Log metrics to MLflow
            log_epoch_metrics(epoch, train_loss, val_loss, metrics, current_lr, epoch_time)
            
            
            # Track best model state
            best_val_loss, new_state, patience = update_best_model(
                val_loss, best_val_loss, model, patience, epoch
            )
            if new_state is not None:
                best_state = new_state
                
            # Periodic plotting
            if epoch % 2 == 0 or epoch == 1:
                generate_epoch_plots(
                    model=model,
                    plot_X=plot_X, 
                    plot_Y=plot_Y, 
                    springback=springback,
                    experiment_config=experiment_config,
                    X_train= X_train, 
                    sensor_names=sensor_names,
                    target_feature_names=target_feature_names, 
                    val_losses=val_losses, 
                    process_part=process_part,
                    train_losses=train_losses, 
                    epoch=epoch,
                    y_lim=y_lim, 
                    predictions_out=predictions_out, 
                    train_loss=train_loss, 
                    val_loss=val_loss,
                    best_val_loss= best_val_loss, 
                    predictions_dir=predictions_dir,
                    attention_csv_dir=attention_csv_dir,
                    attention_dir=attention_dir,
                    loss_dir=loss_dir,
                    diff_hist_dir=diff_hist_dir,
                    diff_bar_dir=diff_bar_dir,
                    val_pred_dir=val_pred_dir,
                    val_diff_bar_dir=val_diff_bar_dir,
                    annot_timesteps=annot_timesteps, 
                    mandrel_extraction_annot_timesteps= mandrel_extraction_annot_timesteps,
                    n_samples = n_samples,
                    val_targets=val_targets_denorm,
                    val_preds=val_preds_denorm,
                    target_scaler=target_scaler,
                )
            
            progress_info = format_progress_bar(
                train_loss, val_loss, metrics, best_val_loss, current_lr, patience
            )
            
            # Update progress bar
            fpbar.set_postfix(progress_info, refresh=True)
            
            
            # Early stopping condition
            if patience >= 20:
                mlflow.log_param("stopped_at_epoch", epoch)
                break
        
        
        # Load best-performing model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        
        # Final evaluation
        all_targets, all_preds = evaluate_final_model(model, val_loader, device)
        all_targets = _maybe_denormalize_targets(all_targets, target_scaler)
        all_preds = _maybe_denormalize_targets(all_preds, target_scaler)
        log_final_metrics(all_targets, all_preds, val_losses, epoch_times)
        
        
        # Persist trained model
        mlflow.pytorch.log_model(model.cpu(), "model")
        logger.info("Model training completed and logged to MLflow.")
        
        
        # Generate final metrics plots
        plot_all_metrics(metrics_history, train_losses, val_losses,
                        learning_rates, epoch_times, training_metric_results_dir)
        
        
        # Feature importance and interpretability
        combined_importance_df, _, importance_paths = analyze_feature_importance(
        model=model,
        val_loader=val_loader,
        feature_names=sensor_names,
        saving_dir=feature_importance_dir
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
            
            log_feature_importance_to_mlflow(combined_importance_df)
        
        if params.get("timestep_sensitivity", False):
            run_all_timestep_sensitivity(
                model=model,
                val_loader=val_loader,
                X_sample=plot_X[-1:],
                springback_sample=springback[-1:],
                experiment_config=experiment_config[-1:],
                device=device,
                saving_dir=timestep_sensitivity_dir,
                occluded_window_size=occlusion_params.get("occlusion_window_size", 10),
                stride=occlusion_params.get("occlusion_stride", 5),
                annot_timesteps=annot_timesteps,
                mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
                sensor_data=plot_X[-1].detach().cpu().numpy(),
                sensor_names=sensor_names,
            )

        
        # Final attention visualization
        generate_final_attention_plot(
            model,
            plot_X,
            springback,
            experiment_config,
            X_test,
            sensor_names,
            process_part,
            attention_lines_dir,
            annot_timesteps,
            mandrel_extraction_annot_timesteps,
        )
        
        
        # Window-based occlusion importance (all angles)
        all_importance_data = []
        for n_angle in range(46):
            importance_df, mean_importance = window_based_importance(
                model=model,
                train_loader=val_loader,
                n_angle=n_angle,
                occluded_window_size=occlusion_params.get("occlusion_window_size", 10),
                stride=occlusion_params.get("occlusion_stride", 5),
                device=device
            )

            visualize_window_importance(
                angle=n_angle,
                feature_names=sensor_names,
                mean_importance=mean_importance,
                annot_timesteps=annot_timesteps,
                window_importance_plots_dir=window_importance_plots_dir,
                mandrel_extraction_annot_timesteps=mandrel_extraction_annot_timesteps,
                process_part=process_part,
                occluded_window_size=occlusion_params.get("occlusion_window_size", 10),
                stride=occlusion_params.get("occlusion_stride", 5)
            )
            all_importance_data.append((n_angle, importance_df, mean_importance))

        save_window_importance_results(all_importance_data, window_importance_plots_dir)
 
 
        # Archive all generated figures
        move_images_to_mlflow_artifacts(base_dir)
