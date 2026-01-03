# ============================
# Standard library imports
# ============================
import time
import warnings
from tqdm import tqdm
from pathlib import Path      
import logging

# ============================
# Third-party ML utilities
# ============================
from sklearn.model_selection import train_test_split

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
from src.pipeline.ml.context_extractor.utils.helpers.plot_utils import (
    compute_plot_limits,
    generate_epoch_plots,
    plot_all_metrics,
    generate_final_attention_plot,
    get_plot_batch,
    visualize_window_importance
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



def train_model(
    X: torch.Tensor,
    Y: torch.Tensor,
    params: dict,
    occlusion_params: dict,
    sensor_names: list[str],
    target_feature_names: list[str],
    machine_part: str,
    preprocessing_info: dict,
    annot_timesteps: list[int],
    mandrel_extraction_annot_timesteps: list[int],
) -> None:
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
    machine_part = machine_part

    predictions_dir = base_dir / "01_predictions"
    loss_dir = base_dir / "02_loss"
    attention_dir = base_dir / "03_attention"
    attention_csv_dir = base_dir / "03_attention_csv"
    attention_lines_dir = base_dir / "04_attention_lines"
    window_importance_plots_dir = base_dir / "07_window_importance"


    # Create directories if they do not exist
    for d in [
        predictions_dir, loss_dir, attention_dir,
        attention_csv_dir, attention_lines_dir,
        window_importance_plots_dir
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
            machine_part, params, preprocessing_info, X, Y, target_feature_names
        )
        
        
    # Split dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )


    # Create PyTorch DataLoaders using customized dataloader
    train_loader, val_loader, plot_loader = create_data_loaders(
        X_train, Y_train, X_val, Y_val, params["batch_size"]
    )


    # Extract a fixed batch for visualization
    plot_X, plot_Y = get_plot_batch(plot_loader, device)
    n_samples = min(4, len(plot_Y))
    
    
    # Compute y-axis limits for consistent plotting
    y_lim = compute_plot_limits(Y_val)

    # ============================================================
    # CASE 1 — TRAIN = FALSE → RESUME EXISTING RUN
    # ============================================================
    if not params.get("train"):
        run_id, model_uri = find_previous_mlflow_run(
            machine_part, preprocessing_info
        )

        if run_id is None:
            logger.warning("No matching MLflow run found to resume.")
            return

        logger.info(f"Resuming MLflow run_id={run_id}")

        with mlflow.start_run(run_id=run_id):
            # Load previously trained model
            model = mlflow.pytorch.load_model(model_uri, map_location=device)


            # Generate final attention visualization
            generate_final_attention_plot(
                model, plot_X, X_val, sensor_names, machine_part, attention_lines_dir,
                annot_timesteps, mandrel_extraction_annot_timesteps
            )


            # Compute integrated gradients if importance is available
            combined_importance_df, _, importance_paths = analyze_feature_importance(
                model=model,
                val_loader=val_loader,
                feature_names=sensor_names
            )

            if combined_importance_df is not None:
                X_sample = plot_X[:1]
                sensor_data_sample = X_val[:1].cpu().numpy()
                save_integrated_gradients_combined(
                    model, X_sample, sensor_data_sample, sensor_names,
                    target_feature_names, base_dir, machine_part, annot_timesteps,
                    mandrel_extraction_annot_timesteps
                )
                
                log_feature_importance_to_mlflow(
                    combined_importance_df
                )
               
               
            # Window-based occlusion importance analysis 
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
                machine_part=machine_part,
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
    with mlflow.start_run(run_name=f"{machine_part}_{excluded58}_ws{window_size}"):
        # Log experiment metadata
        save_experiment_description_as_text(experiment_desc)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        
        y_lim = compute_plot_limits(Y_val)
        plot_X, plot_Y = get_plot_batch(plot_loader, device)
        n_samples = min(4, len(plot_Y))
        
        
        # Initialize model
        model = create_model(features_in, predictions_out, features_out,
                           params["hidden_dim"], params["lstm_layers"],
                           params["dropout"], device)
        
        # Log model architecture parameters
        log_model_parameters(model)
        
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        
        # Loss function
        criterion = nn.MSELoss()
        
        
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
            
            
            # Train and validate
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_preds, val_targets = validate_one_epoch(
                model, val_loader, criterion, device
            )
            
            
            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            
            # Compute evaluation metrics
            metrics = compute_epoch_metrics(val_targets, val_preds)
            for key in metrics_history.keys():
                metrics_history[key].append(metrics[key])
            
            
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
                    X_train= X_train, 
                    sensor_names=sensor_names,
                    target_feature_names=target_feature_names, 
                    val_losses=val_losses, 
                    machine_part=machine_part,
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
                    annot_timesteps=annot_timesteps, 
                    mandrel_extraction_annot_timesteps= mandrel_extraction_annot_timesteps,
                    n_samples = n_samples
                )
            
            progress_info = format_progress_bar(
                train_loss, val_loss, metrics, best_val_loss, current_lr, patience
            )
            
            # Update progress bar
            fpbar.set_postfix(progress_info, refresh=True)
            
            
            # Early stopping condition
            if patience >= 10:
                mlflow.log_param("stopped_at_epoch", epoch)
                break
        
        
        # Load best-performing model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        
        # Final evaluation
        all_targets, all_preds = evaluate_final_model(model, val_loader, device)
        log_final_metrics(all_targets, all_preds, val_losses, epoch_times)
        
        
        # Persist trained model
        mlflow.pytorch.log_model(model.cpu(), "model")
        logger.info("Model training completed and logged to MLflow.")
        
        
        # Generate final diagnostic plots
        plot_all_metrics(metrics_history, train_losses, val_losses,
                        learning_rates, epoch_times, base_dir)
        
        
        # Feature importance and interpretability
        combined_importance_df, _, importance_paths = analyze_feature_importance(
        model=model,
        val_loader=val_loader,
        feature_names=sensor_names
    )
        
        if combined_importance_df is not None:
            X_sample = plot_X[:1]
            sensor_data_sample = X_val[:1].cpu().numpy()
            save_integrated_gradients_combined(
                model, X_sample, sensor_data_sample, sensor_names,
                target_feature_names, base_dir, machine_part, annot_timesteps,
                mandrel_extraction_annot_timesteps
            )
            
            log_feature_importance_to_mlflow(combined_importance_df)
        
        
        # Final attention visualization
        generate_final_attention_plot(
            model, plot_X, X_val, sensor_names, machine_part, attention_lines_dir,
            annot_timesteps, mandrel_extraction_annot_timesteps
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
                machine_part=machine_part,
                occluded_window_size=occlusion_params.get("occlusion_window_size", 10),
                stride=occlusion_params.get("occlusion_stride", 5)
            )
            all_importance_data.append((n_angle, importance_df, mean_importance))

        save_window_importance_results(all_importance_data, window_importance_plots_dir)
 
 
        # Archive all generated figures
        move_images_to_mlflow_artifacts(base_dir)