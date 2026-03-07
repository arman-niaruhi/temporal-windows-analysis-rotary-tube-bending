import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import os
import logging
import json
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from src.pipeline.ml.classification.utils.model import LSTMSequenceClassifier
from src.pipeline.ml.classification.utils.dataset_utils import (
    SegmentDataset3DSequenceWithMask,
)
from src.pipeline.preprocessing.loader import DataLoader as DataLoaderETL
from src.pipeline.ml.classification.utils.preprocessing_utils import (
    ClassifierPreprocessor,
)

logger = logging.getLogger(__name__)


TEST_EXPERIMENT_IDS = [
    2, 3, 22, 23, 40, 54, 83, 85, 110, 112, 119, 120, 121, 122, 123,
    178, 179, 182, 183, 211, 212, 213, 255, 258, 261, 271, 272, 273,
    302, 303, 304, 317, 318
]

PREFERRED_LABEL_ORDER = [
    "Clamping",
    "Bending",
    "Mandrel Extraction",
    "De-Clamping",
]


def _save_confusion_matrix_plot(
    cm: np.ndarray,
    axis_labels: list[str],
    title: str,
    output_file: Path,
) -> None:
    n_classes = len(axis_labels)
    fig_w = max(7.0, 1.8 * n_classes)
    fig_h = max(6.0, 1.4 * n_classes)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_positions = np.arange(len(axis_labels))
    ax.set(
        xticks=tick_positions,
        yticks=tick_positions,
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    x_tick_labels = [label.replace(" ", "\n") for label in axis_labels]
    ax.set_xticklabels(x_tick_labels)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        pad=16,
        labelsize=12,
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", va="bottom")
    ax.tick_params(axis="y", labelsize=14)
    ax.xaxis.label.set_size(18)
    ax.xaxis.labelpad = 18
    ax.yaxis.label.set_size(18)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_validation_confusion_matrices(
    database_path: str,
    annotation_json_path: str,
    experiment_ids_path: str,
    eliminated_columns: list[str],
    models_path: str,
    model_config: dict,
    labels: list[str],
    process_part: str,
    save_dir_path: Any,
    batch_size: int = 8,
) -> None:
    """Save one validation-set confusion matrix per label into inference output dir."""
    output_dir = Path(save_dir_path) / process_part / "All_in_One"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(experiment_ids_path, "r") as f:
        experiment_groups = json.load(f)

    loader = DataLoaderETL(database_path)
    dataframes = loader.load_all_data_from_sqlite()
    base_df = dataframes[process_part]

    ordered_labels = [lbl for lbl in PREFERRED_LABEL_ORDER if lbl in labels]
    ordered_labels += [lbl for lbl in labels if lbl not in PREFERRED_LABEL_ORDER]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = model_config["hidden_size"]
    num_layers = model_config["num_layers"]

    for label in ordered_labels:
        classifier_preprocessor = ClassifierPreprocessor(
            sensors_df=base_df.copy(), annotation_json=annotation_json_path
        )
        sensors_df, _ = classifier_preprocessor.read_data()
        sensors_df = classifier_preprocessor.delete_columns(
            eliminated_columns=eliminated_columns
        )

        cols_to_normalize = sensors_df.columns.drop("Experiment_ID")
        sensors_df[cols_to_normalize] = (
            sensors_df[cols_to_normalize] - sensors_df[cols_to_normalize].min()
        ) / (
            sensors_df[cols_to_normalize].max() - sensors_df[cols_to_normalize].min()
        )
        sensors_df = classifier_preprocessor.assign_one_label(target_label=label)
        sensors_df = classifier_preprocessor.normalize_and_encode_labels()
        feature_cols = classifier_preprocessor.get_feature_cols()

        _, val_df, _, _ = classifier_preprocessor.split_experiments(experiment_groups)
        if val_df.empty:
            logger.warning("Validation set is empty for label %s. Skipping.", label)
            continue

        val_dataset = SegmentDataset3DSequenceWithMask(val_df, feature_cols)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model_path = os.path.join(models_path, process_part, label, "activity_detector.pth")
        try:
            state_dict = torch.load(model_path, map_location=device)
        except FileNotFoundError:
            logger.warning("Model file not found for label %s at %s", label, model_path)
            continue

        num_classes = state_dict["fc.weight"].shape[0]
        model = LSTMSequenceClassifier(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            bidirectional=True,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        metrics = model.evaluate_loader(val_loader, device)
        y_true = metrics.get("y_true", [])
        y_pred = metrics.get("y_pred", [])
        if not y_true or not y_pred:
            logger.warning("No validation predictions for label %s. Skipping.", label)
            continue

        safe_label = label.replace(" ", "_").replace("/", "_")
        class_indices = sorted(val_dataset.label_to_idx.values())
        idx_to_label = {v: k for k, v in val_dataset.label_to_idx.items()}
        axis_labels = [idx_to_label[idx] for idx in class_indices]
        cm = confusion_matrix(y_true, y_pred, labels=class_indices)
        target = output_dir / f"00_validation_confusion_matrix_{safe_label}.png"
        _save_confusion_matrix_plot(
            cm=cm,
            axis_labels=axis_labels,
            title=f"{label} CM - Validation",
            output_file=target,
        )

    all_label = "All"
    all_model_path = os.path.join(
        models_path, process_part, all_label, "activity_detector.pth"
    )
    if not Path(all_model_path).exists():
        logger.warning(
            "All_and_One validation confusion matrix skipped: model not found at %s",
            all_model_path,
        )
        return

    classifier_preprocessor = ClassifierPreprocessor(
        sensors_df=base_df.copy(), annotation_json=annotation_json_path
    )
    sensors_df, _ = classifier_preprocessor.read_data()
    sensors_df = classifier_preprocessor.delete_columns(
        eliminated_columns=eliminated_columns
    )
    cols_to_normalize = sensors_df.columns.drop("Experiment_ID")
    sensors_df[cols_to_normalize] = (
        sensors_df[cols_to_normalize] - sensors_df[cols_to_normalize].min()
    ) / (
        sensors_df[cols_to_normalize].max() - sensors_df[cols_to_normalize].min()
    )
    sensors_df = classifier_preprocessor.assign_labels()
    sensors_df = classifier_preprocessor.normalize_and_encode_labels()
    feature_cols = classifier_preprocessor.get_feature_cols()
    _, val_df, _, _ = classifier_preprocessor.split_experiments(experiment_groups)
    if val_df.empty:
        logger.warning("Validation set is empty for All_and_One. Skipping.")
        return

    val_dataset = SegmentDataset3DSequenceWithMask(val_df, feature_cols)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    state_dict = torch.load(all_model_path, map_location=device)
    num_classes = state_dict["fc.weight"].shape[0]
    all_model = LSTMSequenceClassifier(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        bidirectional=True,
    ).to(device)
    all_model.load_state_dict(state_dict)
    all_model.eval()

    all_metrics = all_model.evaluate_loader(val_loader, device)
    all_y_true = all_metrics.get("y_true", [])
    all_y_pred = all_metrics.get("y_pred", [])
    if not all_y_true or not all_y_pred:
        logger.warning("No validation predictions for All_and_One. Skipping.")
        return

    all_class_indices = sorted(val_dataset.label_to_idx.values())
    all_idx_to_label = {v: k for k, v in val_dataset.label_to_idx.items()}
    all_axis_labels = [all_idx_to_label[idx] for idx in all_class_indices]
    all_cm = confusion_matrix(all_y_true, all_y_pred, labels=all_class_indices)
    all_target = output_dir / "00_validation_confusion_matrix_All_and_One.png"
    _save_confusion_matrix_plot(
        cm=all_cm,
        axis_labels=all_axis_labels,
        title="All_and_One CM - Validation",
        output_file=all_target,
    )


def get_all_predictions(
    label: str,
    database_path: str,
    annotation_json_path: str,
    eliminated_columns: list[str],
    models_path: str,
    process_part: str,
    model_config: dict,
    exp_id: int,
):
    """Get all predictions for a specific experiment.

    Args:
        Label: Target label for which predictions are made.
        database_path: Path to the SQLite database.
        annotation_json_path: Path to the JSON file with annotations.
        eliminated_columns: List of columns to eliminate from the data.
        models_path: Path to the directory containing trained models.
        process_part: Machine part identifier.
        exp_id: Experiment ID to get predictions for.
    Returns:
        exp_data: DataFrame with sensor data for the experiment.
    """
    loader = DataLoaderETL(database_path)
    dataframes = loader.load_all_data_from_sqlite()
    classifier_preprocessor = ClassifierPreprocessor(
        sensors_df=dataframes[process_part], annotation_json=annotation_json_path
    )
    sensors_df, _ = classifier_preprocessor.read_data()
    sensors_df = classifier_preprocessor.delete_columns(
        eliminated_columns=eliminated_columns
    )
    # Match training preprocessing: min-max normalize all sensor features.
    cols_to_normalize = sensors_df.columns.drop("Experiment_ID")
    sensors_df[cols_to_normalize] = (
        sensors_df[cols_to_normalize] - sensors_df[cols_to_normalize].min()
    ) / (
        sensors_df[cols_to_normalize].max() - sensors_df[cols_to_normalize].min()
    )
    sensors_df = classifier_preprocessor.assign_one_label(target_label=label)
    sensors_df = classifier_preprocessor.normalize_and_encode_labels()
    feature_cols = classifier_preprocessor.get_feature_cols()
    input_size = len(feature_cols) 
    hidden_size = model_config["hidden_size"]
    num_layers = model_config["num_layers"]
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSequenceClassifier(
        input_size, hidden_size, num_layers, num_classes, bidirectional=True
    ).to(device)

    model_path = os.path.join(models_path, process_part, label, "activity_detector.pth")

    try:
        state_dict = torch.load(model_path, map_location=device)

    except FileNotFoundError:
        logger.warning(
            f"Model file not found at {model_path}. Please ensure the model has been trained and the path is correct."
        )
        return None, None, None

    model.load_state_dict(state_dict)
    model.eval()

    exp_data = sensors_df[sensors_df["Experiment_ID"] == str(exp_id)]
    X = torch.tensor(exp_data[feature_cols].values, dtype=torch.float32).to(device)
    X = X.unsqueeze(0)  

    with torch.no_grad():
        outputs = model(X)
        y_pred = torch.argmax(outputs, dim=-1).squeeze(0).cpu().numpy()

    mask_decalmping_pred = y_pred == 0
    mask_declamping_true = exp_data["Label_encoded"].values == 0
    return exp_data, mask_decalmping_pred, mask_declamping_true


def inference_one_label_in_one(
    exp_id: int,
    database_path: str,
    annotation_json_path: str,
    eliminated_columns: list[str],
    models_path: str,
    model_config: dict,
    labels: list[str],
    process_part: str,
    save_dir_path: Any,
    get_all_predictions_fn: Any,  
    figsize=(15, 10),
):
    """
    Plot sensor signals and true/predicted process segments for a single experiment.

    Args:
        exp_id: Experiment ID to plot.
        database_path: Path to the SQLite database.
        annotation_json_path: Path to the JSON file with annotations.
        eliminated_columns: List of columns to eliminate from the data.
        models_path: Path to the directory containing trained models.
        labels: List of target labels to plot.
        process_part: Machine part identifier.
        get_all_predictions_fn: Function to get all predictions for a label.
        figsize: Figure size for the plot.

    Returns:
        None
    """
    
    if not isinstance(exp_id, int):
        raise TypeError("exp_id must be an int")

    if not isinstance(database_path, str) or not database_path:
        raise TypeError("database_path must be a non-empty string")
    if not Path(database_path).exists():
        raise ValueError(f"database_path not found: {database_path}")

    if not isinstance(annotation_json_path, str) or not annotation_json_path:
        raise TypeError("annotation_json_path must be a non-empty string")
    if not Path(annotation_json_path).exists():
        raise ValueError(f"annotation_json_path not found: {annotation_json_path}")

    if not isinstance(eliminated_columns, (list, tuple)):
        raise TypeError("eliminated_columns must be a list or tuple of column names")
    if not all(isinstance(c, str) for c in eliminated_columns):
        raise TypeError("all eliminated_columns must be strings")

    if not isinstance(models_path, str) or not models_path:
        raise TypeError("models_path must be a non-empty string")
    if not Path(models_path).exists():
        raise ValueError(f"models_path not found: {models_path}")

    if not isinstance(labels, (list, tuple)) or len(labels) == 0:
        raise ValueError("labels must be a non-empty list of label names")
    if not all(isinstance(l, str) for l in labels):  
        raise TypeError("all labels must be strings")

    if not isinstance(process_part, str) or not process_part:
        raise TypeError("process_part must be a non-empty string")

    if save_dir_path is None:
        raise ValueError("save_dir_path must be provided")
    save_dir_path = Path(save_dir_path)

    if not callable(get_all_predictions_fn):
        raise TypeError("get_all_predictions_fn must be callable")

    logger.info(f"Starting inference for Experiment ID: {exp_id} with labels: {labels}")

    ordered_labels = [lbl for lbl in PREFERRED_LABEL_ORDER if lbl in labels]
    ordered_labels += [lbl for lbl in labels if lbl not in PREFERRED_LABEL_ORDER]

    plot_labels = ordered_labels.copy()
    base_colors = list(mcolors.TABLEAU_COLORS.values())[: len(plot_labels)] 
    
    all_data = {}
    for label in ordered_labels:
        exp_data, mask_pred, mask_true = get_all_predictions_fn(
            label=label,
            database_path=database_path,
            annotation_json_path=annotation_json_path,
            eliminated_columns=eliminated_columns,
            models_path=models_path,
            model_config=model_config,
            process_part=process_part,
            exp_id=exp_id,
        )   # type: ignore
        all_data[label] = {
            "exp_data": exp_data,
            "mask_pred": mask_pred,
            "mask_true": mask_true,
        }
    fig_width = 16  
    fig_height = 5  
    figsize = (fig_width, fig_height)

    _, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1])  

    sensor_plotted = set()
    for label in ordered_labels:
        exp_data = all_data[label]["exp_data"]
        sensor_cols = [
            col
            for col in exp_data.columns
            if col not in ["Label", "Label_encoded", "Experiment_ID"]
        ]

        for col in sensor_cols:
            if col not in sensor_plotted:
                axs[0].plot(exp_data[col].values, linewidth=.8, label=col)
                sensor_plotted.add(col)
            else:
                axs[0].plot(exp_data[col].values, linewidth=.8)

    axs[0].set_ylabel("Sensor Value")
    axs[0].set_title(f"Sensor Signals – Experiment {exp_id}")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")

    axs[0].margins(x=0.01)  

    categories, starts, ends, colors = [], [], [], []

    for i, label in enumerate(plot_labels):
        mask_true = all_data[label]["mask_true"]
        mask_pred = all_data[label]["mask_pred"]
        if label == "Bending" and "Mandrel Extraction" in all_data:
            mandrel_true = all_data["Mandrel Extraction"]["mask_true"]
            mandrel_pred = all_data["Mandrel Extraction"]["mask_pred"]
            mask_true = mask_true | mandrel_true
            mask_pred = mask_pred | mandrel_pred

        base_color = base_colors[i]
        true_color = mcolors.to_rgba(base_color, alpha=0.7)
        pred_color = mcolors.to_rgba(base_color, alpha=0.3)

        def extract_segments(mask, category, color):
            start = None
            for j, val in enumerate(mask):
                if val == 1 and start is None:
                    start = j
                elif val == 0 and start is not None:
                    categories.append(category)
                    starts.append(start)
                    ends.append(j - 1)
                    colors.append(color)
                    start = None
            if start is not None:
                categories.append(category)
                starts.append(start)
                ends.append(len(mask) - 1)
                colors.append(color)

        extract_segments(mask_true, f"{label} True", true_color)
        extract_segments(mask_pred, f"{label} Pred", pred_color)

    if categories:
        axs[1].barh(
            categories,
            [e - s for s, e in zip(starts, ends)],
            left=starts,
            color=colors,
            height=0.6,
        )
    else:
        axs[1].text(
            0.5,
            0.5,
            "No Segments Found",
            transform=axs[1].transAxes,
            ha="center",
            va="center",
        )

    axs[1].set_xlabel("Time Index")
    axs[1].margins(x=0.01)

    plt.tight_layout()
    output_dir = save_dir_path / process_part /"All_in_One"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"individual_labels_{exp_id}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
