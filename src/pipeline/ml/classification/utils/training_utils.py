import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.pipeline.ml.classification.utils.preprocessing_utils import (
    ClassifierPreprocessor,
)
from src.pipeline.ml.classification.utils.dataset_utils import (
    SegmentDataset3DSequenceWithMask,
)
from src.pipeline.ml.classification.utils.feature_analysis_utils import (
    compare_methods,
    dropout_importance,
    feature_ablation_importance,
    gradient_importance_sequence,
    integrated_gradients_importance,
    occlusion_importance,
    permutation_importance_sequence,
    plot_feature_importance,
)
from src.pipeline.ml.classification.utils.model import LSTMSequenceClassifier
from src.pipeline.preprocessing.loader import DataLoader as DataLoaderETL

logger = logging.getLogger(__name__)

# Constants
VALID_LABELS = ["All", "Clamping", "Bending", "Mandrel Extraction", "De-Clamping"]
VALID_MACHINE_PARTS = ["machine_and_movement", "movement"]
EXCLUDED_COLUMNS = ["Experiment_ID", "Label", "Label_encoded"]


def _validate_inputs(
    annotation_json_path: str,
    database_path: str,
    experiment_ids_path: str,
    label: str,
    machine_part: str,
    pipeline_config: Optional[Dict],
) -> None:
    """Validate all input parameters before processing.

    Args:
        annotation_json_path: Path to annotation JSON file
        database_path: Path to SQLite database
        experiment_ids_path: Path to experiment IDs file
        label: Target activity label
        machine_part: Machine part identifier
        pipeline_config: Pipeline configuration dictionary

    Raises:
        ValueError: If any validation check fails
    """
    if not os.path.exists(annotation_json_path):
        raise ValueError(
            f"Annotation JSON file not found: {annotation_json_path}. "
            "Please use annotation.py to create annotations."
        )

    if not os.path.exists(database_path):
        raise ValueError(
            f"Database file not found: {database_path}. "
            "Please run preprocessing first: 'python main.py preprocess'"
        )

    if not os.path.exists(experiment_ids_path):
        raise ValueError(
            f"Experiment IDs file not found: {experiment_ids_path}. "
            "Please download from the GitHub page."
        )

    if label not in VALID_LABELS:
        raise ValueError(
            f"Invalid label: {label}. Valid options: {', '.join(VALID_LABELS)}"
        )

    if machine_part not in VALID_MACHINE_PARTS:
        raise ValueError(
            f"Invalid machine part: {machine_part}. "
            f"Valid options: {', '.join(VALID_MACHINE_PARTS)}"
        )

    if pipeline_config is None:
        raise ValueError("Pipeline configuration is required.")


def _load_experiment_groups(experiment_ids_path: str) -> Dict:
    """Load experiment groups from JSON file.

    Args:
        experiment_ids_path: Path to experiment IDs JSON file

    Returns:
        Dictionary containing experiment groups
    """
    logger.info("Loading experiment IDs...")
    with open(experiment_ids_path, "r") as f:
        return json.load(f)


def _prepare_data(
    database_path: str,
    annotation_json_path: str,
    machine_part: str,
    eliminated_columns: List[str],
    label: str,
    experiment_groups: Dict,
) -> Tuple:
    """Prepare and preprocess data for training.

    Args:
        database_path: Path to SQLite database
        annotation_json_path: Path to annotations
        machine_part: Machine part to process
        eliminated_columns: Columns to remove
        label: Target label
        experiment_groups: Experiment ID groupings

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, feature_cols)
    """
    loader = DataLoaderETL(database_path)
    dataframes = loader.load_all_data_from_sqlite()

    classifier_preprocessor = ClassifierPreprocessor(
        sensors_df=dataframes[machine_part], annotation_json=annotation_json_path
    )

    sensors_df, _ = classifier_preprocessor.read_data()
    sensors_df = classifier_preprocessor.delete_columns(eliminated_columns)

    if label == "All":
        sensors_df = classifier_preprocessor.assign_labels()
    else:
        sensors_df = classifier_preprocessor.assign_one_label(label)

    sensors_df = classifier_preprocessor.normalize_and_encode_labels()

    train_df, val_df, test_df, _ = classifier_preprocessor.split_experiments(
        experiment_groups
    )

    train_dataset, val_dataset, test_dataset = classifier_preprocessor.create_datasets(
        train_df, val_df, test_df, SegmentDataset3DSequenceWithMask
    )

    feature_cols = classifier_preprocessor.get_feature_cols()

    return train_dataset, val_dataset, test_dataset, feature_cols, sensors_df


def _create_data_loaders(
    train_dataset, val_dataset, test_dataset, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and test sets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for data loaders

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def _initialize_model(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    num_classes: int,
    device: torch.device,
) -> LSTMSequenceClassifier:
    """Initialize LSTM model.

    Args:
        input_size: Number of input features
        hidden_size: Hidden layer size
        num_layers: Number of LSTM layers
        num_classes: Number of output classes
        device: Torch device (CPU/GPU)

    Returns:
        Initialized LSTM model
    """
    model = LSTMSequenceClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        bidirectional=True,
    )
    return model.to(device)


def _train_or_load_model(
    model: LSTMSequenceClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pipeline_config: Dict,
    model_path: str,
    label: str,
    machine_part: str,
    device: torch.device,
    idx_to_label: Dict,
) -> LSTMSequenceClassifier:
    """Train a new model or load existing one.

    Args:
        model: LSTM model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        pipeline_config: Training configuration
        model_path: Path to save/load model
        label: Target label name
        machine_part: Machine part identifier
        device: Torch device
        idx_to_label: Index to label mapping

    Returns:
        Trained or loaded model
    """
    training_config = pipeline_config.get("training_config", {})

    if training_config.get("training", False):
        logger.info("Training new model...")
        model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=training_config.get("num_epochs", 300),
            learning_rate=training_config.get("learning_rate", 1e-5),
            patience=training_config.get("patience", 3),
            device=device,
            idx_to_label=idx_to_label,
            model_path=model_path,
            run_name=label,
            experiment_name=machine_part,
        )
    else:
        logger.info("Loading existing model...")
        model_file = Path(model_path) / "activity_detector.pth"
        state_dict = torch.load(model_file, map_location=device)
        logger.info(f"Loaded model from {model_file}")
        model.load_state_dict(state_dict)
        model.eval()

    return model


def training_pipeline(
    model_path_root: str,
    database_path: str,
    annotation_json_path: str,
    experiment_ids_path: str,
    machine_part: str,
    eliminated_columns: List[str],
    label: str,
    pipeline_config: Dict,
) -> Tuple:
    """Execute training pipeline for machine activity recognition.

    Args:
        model_path_root: Root directory for model storage
        database_path: Path to preprocessed data database
        annotation_json_path: Path to activity annotations
        experiment_ids_path: Path to experiment ID groups
        machine_part: Machine component to analyze
        eliminated_columns: Columns to exclude from training
        label: Target activity label to classify
        pipeline_config: Pipeline configuration parameters

    Returns:
        Tuple of (model, sensors_df, test_loader, device, feature_cols)
    """
    logger.info("Starting machine activity recognition training pipeline...")
    logger.info(f"Label: {label}, Machine part: {machine_part}")

    # Validate inputs
    _validate_inputs(
        annotation_json_path,
        database_path,
        experiment_ids_path,
        label,
        machine_part,
        pipeline_config,
    )

    # Load experiment groups
    experiment_groups = _load_experiment_groups(experiment_ids_path)

    # Prepare data
    train_dataset, val_dataset, test_dataset, feature_cols, sensors_df = _prepare_data(
        database_path,
        annotation_json_path,
        machine_part,
        eliminated_columns,
        label,
        experiment_groups,
    )

    # Create data loaders
    dataloader_config = pipeline_config.get("dataloader_config", {})
    batch_size = dataloader_config.get("batch_size", 8)
    train_loader, val_loader, test_loader = _create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )

    # Model configuration
    model_config = pipeline_config.get("model_config", {})
    input_size = len(feature_cols)
    hidden_size = model_config.get("hidden_size", 64)
    num_layers = model_config.get("num_layers", 2)
    num_classes = len(train_dataset.unique_labels)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _initialize_model(input_size, hidden_size, num_layers, num_classes, device)

    # Train or load model
    idx_to_label = {v: k for k, v in train_dataset.label_to_idx.items()}
    model_path = f"{model_path_root}/{machine_part}/{label}"
    model = _train_or_load_model(
        model,
        train_loader,
        val_loader,
        pipeline_config,
        model_path,
        label,
        machine_part,
        device,
        idx_to_label,
    )

    return model, sensors_df, test_loader, device, feature_cols


def _compute_importance_scores(
    model: LSTMSequenceClassifier,
    test_loader: DataLoader,
    device: torch.device,
    result_path: str,
    feature_names: List[str],
) -> Dict[str, np.ndarray]:
    """Compute feature importance using multiple methods.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Torch device
        result_path: Path to save results
        feature_names: List of feature names

    Returns:
        Dictionary mapping method names to importance scores
    """
    all_importances = {}

    # Permutation importance
    logger.info("Computing permutation importance...")
    perm_importances, perm_std = permutation_importance_sequence(
        model, test_loader, device, n_repeats=10
    )
    all_importances["Permutation"] = perm_importances
    plot_feature_importance(
        result_path, perm_importances, perm_std, feature_names, "Permutation"
    )

    # Gradient importance
    logger.info("Computing gradient importance...")
    grad_importances = gradient_importance_sequence(
        model, test_loader, device, n_batches=10
    )
    all_importances["Gradient"] = grad_importances
    plot_feature_importance(
        result_path, grad_importances, None, feature_names, "Gradient"
    )

    # Integrated gradients
    logger.info("Computing integrated gradients...")
    intgrad_importances = integrated_gradients_importance(
        model, test_loader, device, n_samples=50, steps=30
    )
    all_importances["Integrated Gradients"] = intgrad_importances
    plot_feature_importance(
        result_path, intgrad_importances, None, feature_names, "IntegratedGradients"
    )

    # Occlusion importance
    logger.info("Computing occlusion importance...")
    occlusion_importances = occlusion_importance(
        model, test_loader, device, occlusion_value=0.0
    )
    all_importances["Occlusion"] = occlusion_importances
    plot_feature_importance(
        result_path, occlusion_importances, None, feature_names, "Occlusion"
    )

    # Ablation importance
    logger.info("Computing ablation importance...")
    ablation_importances = feature_ablation_importance(model, test_loader, device)
    all_importances["Ablation"] = ablation_importances
    plot_feature_importance(
        result_path, ablation_importances, None, feature_names, "Ablation"
    )

    # Dropout importance
    logger.info("Computing dropout importance...")
    dropout_importances = dropout_importance(
        model, test_loader, device, n_repeats=20, dropout_rate=0.5
    )
    all_importances["Dropout"] = dropout_importances
    plot_feature_importance(
        result_path, dropout_importances, None, feature_names, "Dropout"
    )

    return all_importances


def analyze_features(
    analyze_features_result_path: str,
    model: LSTMSequenceClassifier,
    sensors_df,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    """Analyze feature importance using multiple methods.

    Implements six feature importance methods:
    1. Permutation Importance
    2. Gradient Importance
    3. Integrated Gradients
    4. Occlusion Importance
    5. Feature Ablation
    6. Dropout Importance

    Args:
        analyze_features_result_path: Directory to save results
        model: Trained LSTM model
        sensors_df: Preprocessed sensor DataFrame
        test_loader: Test data loader
        device: Torch device (CPU/GPU)
    """
    logger.info("Starting feature importance analysis...")

    # Extract feature names
    feature_names = [col for col in sensors_df.columns if col not in EXCLUDED_COLUMNS]

    # Compute importance scores
    all_importances = _compute_importance_scores(
        model, test_loader, device, analyze_features_result_path, feature_names
    )

    # Compare methods
    logger.info("Comparing importance methods...")
    compare_methods(all_importances, feature_names)

    # Save all results
    output_file = (
        Path(analyze_features_result_path) / "feature_importance_all_methods.npz"
    )
    logger.info(f"Saving results to {output_file}")

    np.savez(
        output_file,
        permutation_importance=all_importances["Permutation"],
        permutation_std=all_importances.get("Permutation_std"),
        gradient_importance=all_importances["Gradient"],
        integrated_gradients_importance=all_importances["Integrated Gradients"],
        occlusion_importance=all_importances["Occlusion"],
        ablation_importance=all_importances["Ablation"],
        dropout_importance=all_importances["Dropout"],
        feature_names=feature_names,
    )

    logger.info("Feature importance analysis completed.")
