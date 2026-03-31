import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import ast

import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.pipeline.ml.classification.utils.preprocessing_utils import (
    ClassifierPreprocessor,
)
from src.pipeline.ml.classification.utils.dataset_utils import (
    SegmentDataset3DSequenceWithMask,
)
from src.pipeline.ml.classification.utils.model import LSTMSequenceClassifier
from src.pipeline.preprocessing.loader import DataLoader as DataLoaderETL

logger = logging.getLogger(__name__)

VALID_LABELS = ["Multilabel", "Clamping", "Bending", "Mandrel Extraction", "De-Clamping"]
VALID_process_partS = ["machine_and_movement", "movement"]
ELIMINATED_COLUMNS = [
    "PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]",
    "COLLET_ROTATING_Movement_[mm]",
    "BEND-DIE_VERTICAL_Movement_[mm]",
    "PRESSURE-DIE_LATERAL_Movement_[mm]",
]


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _validate_inputs(
    annotation_json_path: str,
    database_path: str,
    experiment_ids_path: str,
    label: str,
    process_part: str,
    pipeline_config: Optional[Dict],
) -> None:
    """Validate all input parameters before processing.

    Args:
        annotation_json_path: Path to annotation JSON file
        database_path: Path to the ETL CSV directory
        experiment_ids_path: Path to experiment IDs file
        label: Target activity label
        process_part: Machine part identifier
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
            f"Processed data path not found: {database_path}. "
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

    if process_part not in VALID_process_partS:
        raise ValueError(
            f"Invalid machine part: {process_part}. "
            f"Valid options: {', '.join(VALID_process_partS)}"
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
    

    df = pd.read_csv(experiment_ids_path)

    experiment_numbers = (
        df["Experiment_Number"]
        .apply(ast.literal_eval)
        .tolist()
    )
    return experiment_numbers


def _prepare_data(
    database_path: str,
    annotation_json_path: str,
    process_part: str,
    label: str,
    experiment_groups: Dict,
    random_seed: int,
) -> Tuple:
    """Prepare and preprocess data for training.

    Args:
        database_path: Path to the ETL CSV directory
        annotation_json_path: Path to annotations
        process_part: Machine part to process
        label: Target label
        experiment_groups: Experiment ID groupings

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, feature_cols)
    """
    loader = DataLoaderETL(database_path)
    dataframes = loader.load_all_data_from_csv()

    classifier_preprocessor = ClassifierPreprocessor(
        sensors_df=dataframes[process_part], annotation_json=annotation_json_path
    )
    sensors_df, _ = classifier_preprocessor.read_data()
    sensors_df = classifier_preprocessor.delete_columns(ELIMINATED_COLUMNS)

    # Exclude identifier column
    cols_to_normalize = sensors_df.columns.drop('Experiment_ID')

    sensors_df[cols_to_normalize] = (
        sensors_df[cols_to_normalize] - sensors_df[cols_to_normalize].min()
    ) / (
        sensors_df[cols_to_normalize].max() - sensors_df[cols_to_normalize].min()
    )

    if label == "Multilabel":
        sensors_df = classifier_preprocessor.assign_labels()
    else:
        sensors_df = classifier_preprocessor.assign_one_label(label)

    sensors_df = classifier_preprocessor.normalize_and_encode_labels()

    train_df, val_df, test_df, _ = classifier_preprocessor.split_experiments(
        experiment_groups,
        seed=random_seed,
    )

    train_dataset, val_dataset, test_dataset = classifier_preprocessor.create_datasets(
        train_df, val_df, test_df, SegmentDataset3DSequenceWithMask
    )

    feature_cols = classifier_preprocessor.get_feature_cols()

    return train_dataset, val_dataset, test_dataset, feature_cols, sensors_df


def _create_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int,
    random_seed: int,
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
    train_generator = torch.Generator().manual_seed(random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def _write_evaluation_report(report_path: Path, metrics_by_split: Dict[str, Dict]) -> None:
    """Persist evaluation metrics for train/val/test/all splits."""
    serializable_metrics = {
        split: {
            key: value
            for key, value in metrics.items()
            if key not in {"y_true", "y_pred"}
        }
        for split, metrics in metrics_by_split.items()
    }

    with open(report_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)


def _evaluate_and_store_metrics(
    model: LSTMSequenceClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    all_loader: DataLoader,
    model_path: str,
    device: torch.device,
) -> Dict[str, Dict]:
    """Evaluate the trained model on all relevant splits and store metrics."""
    metrics_by_split = {
        "train": model.evaluate_loader(train_loader, device),
        "val": model.evaluate_loader(val_loader, device),
        "test": model.evaluate_loader(test_loader, device),
        "all_data": model.evaluate_loader(all_loader, device),
    }

    for split_name, metrics in metrics_by_split.items():
        logger.info(
            "%s metrics | loss: %.6f | acc: %.6f | precision: %.6f | recall: %.6f | f1: %.6f | iou: %.6f",
            split_name,
            metrics["loss"],
            metrics["acc"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["iou"],
        )

    report_path = Path(model_path) / "evaluation_metrics.json"
    _write_evaluation_report(report_path, metrics_by_split)
    logger.info("Saved evaluation metrics to %s", report_path)

    return metrics_by_split


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
    process_part: str,
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
        process_part: Machine part identifier
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
            experiment_name=process_part,
            save_confusion_every=20,
            scheduler_factor=0.5,
            scheduler_patience=2,
            min_lr=1e-6,
        )
    else:
        logger.info("Loading existing model...")
        model_file = Path(model_path) / "activity_detector.pth"
        state_dict = torch.load(model_file, map_location=device, weights_only=True)
        logger.info(f"Loaded model from {model_file}")
        model.load_state_dict(state_dict)
        model.eval()

    return model


def training_pipeline(
    model_path_root: str,
    database_path: str,
    annotation_json_path: str,
    experiment_ids_path: str,
    process_part: str,
    label: str,
    pipeline_config: Dict,
    random_seed: int,
) -> Tuple:
    """Execute training pipeline for machine activity recognition.

    Args:
        model_path_root: Root directory for model storage
        database_path: Path to the preprocessed ETL CSV directory
        annotation_json_path: Path to activity annotations
        experiment_ids_path: Path to experiment ID groups
        process_part: Machine component to analyze
        label: Target activity label to classify
        pipeline_config: Pipeline configuration parameters

    Returns:
        Tuple of (model, sensors_df, test_loader, device, feature_cols)
    """
    logger.info("Starting machine activity recognition training pipeline...")
    logger.info(f"Label: {label}, Machine part: {process_part}")
    logger.info("Using random seed: %s", random_seed)

    set_global_seed(random_seed)

    _validate_inputs(
        annotation_json_path,
        database_path,
        experiment_ids_path,
        label,
        process_part,
        pipeline_config,
    )

    experiment_groups = _load_experiment_groups(experiment_ids_path)

    train_dataset, val_dataset, test_dataset, feature_cols, sensors_df = _prepare_data(
        database_path,
        annotation_json_path,
        process_part,
        label,
        experiment_groups,
        random_seed,
    )

    dataloader_config = pipeline_config.get("dataloader_config", {})
    batch_size = dataloader_config.get("batch_size", 8)
    train_loader, val_loader, test_loader = _create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, random_seed
    )
    all_dataset = SegmentDataset3DSequenceWithMask(sensors_df, feature_cols)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)

    model_config = pipeline_config.get("model_config", {})
    input_size = len(feature_cols)
    hidden_size = model_config.get("hidden_size", 64)
    num_layers = model_config.get("num_layers", 2)
    num_classes = len(train_dataset.unique_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _initialize_model(input_size, hidden_size, num_layers, num_classes, device)

    idx_to_label = {v: k for k, v in train_dataset.label_to_idx.items()}
    model_path = f"{model_path_root}/{process_part}/{label}"
    model = _train_or_load_model(
        model,
        train_loader,
        val_loader,
        pipeline_config,
        model_path,
        label,
        process_part,
        device,
        idx_to_label,
    )

    _evaluate_and_store_metrics(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        all_loader=all_loader,
        model_path=model_path,
        device=device,
    )

    return model, sensors_df, test_loader, device, feature_cols
