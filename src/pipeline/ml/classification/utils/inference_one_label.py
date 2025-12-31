import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import os
import logging
from pathlib import Path

from src.pipeline.ml.classification.utils.model import LSTMSequenceClassifier
from src.pipeline.preprocessing.loader import DataLoader as DataLoaderETL
from src.pipeline.ml.classification.utils.preprocessing_utils import (
    ClassifierPreprocessor,
)

logger = logging.getLogger(__name__)


def get_all_predictions(
    label: str,
    database_path: str,
    annotation_json_path: str,
    eliminated_columns: list[str],
    models_path: str,
    machine_part: str,
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
        machine_part: Machine part identifier.
        exp_id: Experiment ID to get predictions for.
    Returns:
        exp_data: DataFrame with sensor data for the experiment.
    """
    loader = DataLoaderETL(database_path)
    dataframes = loader.load_all_data_from_sqlite()
    classifier_preprocessor = ClassifierPreprocessor(
        sensors_df=dataframes[machine_part], annotation_json=annotation_json_path
    )
    sensors_df, _ = classifier_preprocessor.read_data()
    sensors_df = classifier_preprocessor.delete_columns(
        eliminated_columns=eliminated_columns
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

    model_path = os.path.join(models_path, machine_part, label, "activity_detector.pth")

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
    machine_part: str,
    save_dir_path: str,
    get_all_predictions_fn: callable,  
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
        machine_part: Machine part identifier.
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

    if not isinstance(machine_part, str) or not machine_part:
        raise TypeError("machine_part must be a non-empty string")

    if save_dir_path is None:
        raise ValueError("save_dir_path must be provided")
    save_dir_path = Path(save_dir_path)

    if not callable(get_all_predictions_fn):
        raise TypeError("get_all_predictions_fn must be callable")

    logger.info(f"Starting inference for Experiment ID: {exp_id} with labels: {labels}")

    base_colors = list(mcolors.TABLEAU_COLORS.values())[: len(labels)] 

    all_data = {}
    for label in labels:
        exp_data, mask_pred, mask_true = get_all_predictions_fn(
            label=label,
            database_path=database_path,
            annotation_json_path=annotation_json_path,
            eliminated_columns=eliminated_columns,
            models_path=models_path,
            model_config=model_config,
            machine_part=machine_part,
            exp_id=exp_id,
        )  
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
    for label in labels:
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

    for i, label in enumerate(labels):
        mask_true = all_data[label]["mask_true"]
        mask_pred = all_data[label]["mask_pred"]

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
    save_dir_path = Path(save_dir_path) / machine_part
    save_dir_path.parent.mkdir(parents=True, exist_ok=True)
    save_file_path = save_dir_path / "individual_labels.png"
    plt.savefig(f"{save_file_path}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
