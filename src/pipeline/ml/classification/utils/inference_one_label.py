import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import os

from src.pipeline.ml.classification.utils.model import LSTMSequenceClassifier
from src.pipeline.preprocessing.loader import DataLoader as DataLoaderETL
from src.pipeline.ml.classification.utils.classification_utils import (
    ClassifierPreprocessor,
)


def get_all_predictions(
    Label: str,
    database_path: str,
    annotation_json_path: str,
    eliminated_columns: list[str],
    models_path: str,
    machine_part: str,
    exp_id: int,
):
    loader = DataLoaderETL(database_path)
    dataframes = loader.load_all_data_from_sqlite()
    classifier_preprocessor = ClassifierPreprocessor(
        sensors_df=dataframes[machine_part], annotation_json=annotation_json_path
    )
    sensors_df, _ = classifier_preprocessor.read_data()
    sensors_df = classifier_preprocessor.delete_columns(
        eliminated_columns=eliminated_columns
    )
    sensors_df = classifier_preprocessor.assign_one_label(target_label=Label)
    sensors_df = classifier_preprocessor.normalize_and_encode_labels()
    feature_cols = classifier_preprocessor.get_feature_cols()
    input_size = len(feature_cols)
    hidden_size = 64
    num_layers = 2
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSequenceClassifier(
        input_size, hidden_size, num_layers, num_classes, bidirectional=True
    ).to(device)

    model_path = os.path.join(models_path, machine_part, Label, "activity_detector.pth")

    state_dict = torch.load(
            model_path,
            map_location=device
        )
    model.load_state_dict(state_dict)
    model.eval()

    # --- Select a random experiment ---
    exp_data = sensors_df[sensors_df["Experiment_ID"] == str(exp_id)]
    # --- Features ---
    X = torch.tensor(exp_data[feature_cols].values, dtype=torch.float32).to(device)
    X = X.unsqueeze(0)  # [1, seq_len, num_features]

    # --- Model predictions ---
    with torch.no_grad():
        outputs = model(X)
        y_pred = torch.argmax(outputs, dim=-1).squeeze(0).cpu().numpy()

    # Example boolean masks
    mask_decalmping_pred = y_pred == 0
    mask_declamping_true = exp_data["Label_encoded"].values == 0
    return exp_data, mask_decalmping_pred, mask_declamping_true


def inference_one_label_in_one(
    exp_id,
    database_path,
    annotation_json_path,
    eliminated_columns,
    models_path,
    labels,
    machine_part,
    get_all_predictions_fn,
    figsize=(15, 10),
):
    """
    Plot sensor signals and true/predicted process segments for a single experiment.

    Parameters
    ----------
    exp_id : int or str
        Experiment ID to plot.
    machine_part : str
        Machine part passed to get_all_predictions.
    labels : list of str
        Ordered list of process labels.
    get_all_predictions_fn : callable
        Function with signature:
        get_all_predictions_fn(Label, machine_part, exp_id)
        -> (exp_data, mask_pred, mask_true)
    figsize : tuple, optional
        Figure size for the plot.
    """

    # Assign distinct base colors
    base_colors = list(mcolors.TABLEAU_COLORS.values())[: len(labels)]

    # --- Load all data upfront ---
    all_data = {}
    for label in labels:
        exp_data, mask_pred, mask_true = get_all_predictions_fn(
            label,
            database_path,
            annotation_json_path,
            eliminated_columns,
            models_path,
            machine_part=machine_part,
            exp_id=exp_id,
        )
        all_data[label] = {
            "exp_data": exp_data,
            "mask_pred": mask_pred,
            "mask_true": mask_true,
        }

    # --- Create figure ---
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # --- Top subplot: sensor signals ---
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
                axs[0].plot(exp_data[col].values, label=col)
                sensor_plotted.add(col)
            else:
                axs[0].plot(exp_data[col].values)

    axs[0].set_ylabel("Sensor Value")
    axs[0].set_title(f"Sensor Signals – Experiment {exp_id}")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")

    # --- Bottom subplot: true vs predicted segments ---
    categories, starts, ends, colors = [], [], [], []

    for i, label in enumerate(labels):
        mask_true = all_data[label]["mask_true"]
        mask_pred = all_data[label]["mask_pred"]

        base_color = base_colors[i]
        true_color = mcolors.to_rgba(base_color, alpha=0.7)
        pred_color = mcolors.to_rgba(base_color, alpha=0.3)

        # Helper to extract segments
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

    plt.tight_layout()
    plt.show()
