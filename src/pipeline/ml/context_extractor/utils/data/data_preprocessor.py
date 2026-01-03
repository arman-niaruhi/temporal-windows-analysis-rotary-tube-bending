import logging
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from src.pipeline.ml.context_extractor.utils.data.data_resampler import (
    resample_experiment_ultrafast,
)
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from src.pipeline.preprocessing.loader import DataLoader as DataLoaderETL
import json

# Initialize logger for monitoring execution
logger = logging.getLogger(__name__)

class LSTMPreprocessor:
    """
    Preprocessor class for LSTM-based time-series modeling of machine sensor data.

    This class handles:
    - Reading raw sensor and target data
    - Feature selection based on variance thresholds
    - Normalization of sensor and target data
    - Grouping and padding of sequences to prepare 3D tensors
    """

    def __init__(self, database_path: str, machine_part: str, annotation_json_path: str) -> None:
        """
        Initialize the preprocessor.

        Args:
            database_path: Path to the SQLite database containing raw data.
            machine_part: Identifier of the machine part to model (e.g., 'De-Clamping').
            annotation_json_path: Path to JSON file containing label annotations.
        """
        self.machine_part = machine_part
        self.sensor_df = None
        self.target_df = None
        self._feature_cols = None
        self.annotation_json_path = annotation_json_path
        self.database_path = database_path

    def read_data(self, label_name: str):
        """
        Load sensor and target data from the database and annotations.

        Args:
            label_name: Label to extract; if "All", all sensor data is returned.

        Returns:
            Tuple of sensor DataFrame and target DataFrame.
        """
        loader = DataLoaderETL(self.database_path)
        dataframes = loader.load_all_data_from_sqlite()
        sensors_df = dataframes["machine_and_movement"]

        with open(self.annotation_json_path, "r") as f:
            labels = json.load(f)

        if label_name == "All":
            sensors_df.index = sensors_df["Time_[s]"]
            sensors_df.drop(columns=["Time_[s]"], inplace=True)
            return sensors_df, dataframes["arc"]

        # Extract label windows and merge with sensor data
        label_windows = pd.DataFrame(
            [
                {
                    "Experiment_ID": int(exp_id),
                    "start": phase["start"],
                    "end": phase["end"],
                }
                for exp_id, phases in labels.items()
                for phase in phases
                if phase["label"] == label_name
            ]
        )
        df_label = (
            sensors_df.merge(label_windows, on="Experiment_ID", how="inner")
            .query("`Time_[s]` >= start and `Time_[s]` <= end")
            .drop(columns=["start", "end"])
        )
        df_label.index = df_label["Time_[s]"]
        df_label.drop(columns=["Time_[s]"], inplace=True)
        logger.info("Data read complete.")
        return df_label, dataframes["arc"]

    def feature_selection(self, variance_threshold: float = 0):
        """
        Select features based on variance threshold to remove low-variance features.

        Args:
            variance_threshold: Minimum variance required for a feature to be retained.

        Returns:
            Filtered sensor DataFrame.
        """
        if self.sensor_df is None:
            logger.warning("sensor_df is None. Cannot perform feature selection.")
            raise FileNotFoundError
        self.sensor_df = self.sensor_df.loc[:, self.sensor_df.var() > variance_threshold]
        return self.sensor_df

    def get_feature_cols(self) -> list[str]:
        """Return the list of feature column names."""
        if not self._feature_cols:
            raise ValueError("Feature columns are not set. Call set_feature_cols() first.")
        return self._feature_cols

    def get_dfs(self):
        """Return sensor and target DataFrames."""
        if self.sensor_df is None:
            raise ValueError("sensor_df is None. Load data first.")
        if self.target_df is None:
            raise ValueError("target_df is None. Load data first.")
        return self.sensor_df, self.target_df

    def group_and_pad(self, df: pd.DataFrame, group_col: str = "Experiment_ID",
                      exclude_cols: Optional[list[str]] = None) -> np.ndarray:
        """
        Group data by experiment and pad sequences to the length of the longest experiment.

        Args:
            df: Input DataFrame.
            group_col: Column to group by (default: 'Experiment_ID').
            exclude_cols: Columns to exclude from padding.

        Returns:
            3D numpy array (num_groups, max_group_length, num_features), NaNs replaced with 0.
        """
        if exclude_cols is None:
            exclude_cols = [group_col]

        groups_list = []
        max_len = df.groupby(group_col).size().max()
        num_features = df.shape[1] - len(exclude_cols)

        for _, group in df.groupby(group_col):
            values = group.drop(columns=exclude_cols).values
            padded = np.full((max_len, num_features), np.nan)
            padded[: values.shape[0], :] = values
            groups_list.append(padded)

        array_3d = np.array(groups_list, dtype=float)
        return np.nan_to_num(array_3d, nan=0.0)

    def normalize_column(self, column_name: str) -> pd.DataFrame:
        """Normalize a specific target column to range [0,1]."""
        if self.target_df is None:
            raise ValueError("target_df is None. Load target data first.")

        if column_name not in self.target_df.columns:
            raise ValueError(
                f"Column '{column_name}' not found in target_df. Available columns: {list(self.target_df.columns)}"
            )
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_df[column_name] = scaler.fit_transform(self.target_df[[column_name]])
        return self.target_df

    def normalize_angle(self, col: str = "Angle[degree]ORDistance[mm]") -> pd.DataFrame:
        """Normalize angle-distance column for sequence modeling."""
        return self.normalize_column(col)


class ProcessDataset(Dataset):
    """Custom PyTorch Dataset for time-series data."""

    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.X = X.float()
        self.Y = Y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def _normalize_experiment(group, n=46):
    """
    Ensure each experiment has exactly n rows by truncating or padding with the last row.

    Args:
        group: DataFrame for one experiment.
        n: Desired number of rows.

    Returns:
        Normalized DataFrame.
    """
    if len(group) > n:
        return group.iloc[:n].copy()
    else:
        return group.copy()


def _scale_annotation_timesteps(
    annot_timesteps: list[int],
    mandrel_extraction_timesteps: list[int],
    sequence_length: int,
    reference_length: int = 1743,
) -> tuple[list[int], list[int]]:
    """
    Scale annotation indices from reference timeline to actual sequence length.

    Args:
        annot_timesteps: Annotation indices on reference timeline.
        mandrel_extraction_timesteps: Mandrel extraction indices on reference timeline.
        sequence_length: Target sequence length after padding/resampling.
        reference_length: Original annotation reference length.

    Returns:
        Tuple of scaled annotation timesteps.
    """
    def _scale(indices: list[int]) -> list[int]:
        return [int((idx / reference_length) * sequence_length) for idx in indices]

    return _scale(annot_timesteps), _scale(mandrel_extraction_timesteps)


def prepare_data(input_path_param: dict, preprocessing_param: dict):
    """
    Main data preparation function for LSTM training.

    Handles:
    - Loading and filtering raw data
    - Resampling sensor and target sequences
    - Selecting target features
    - Grouping and padding into 3D tensors
    - Scaling annotation indices

    Returns:
        X: Input tensor
        Y: Target tensor
        sensor_names: Feature names
        target_feature_names: Target columns
        annot_timesteps: Scaled annotation indices
        mandrel_extraction_annot_timesteps: Scaled mandrel extraction indices
    """
    machine_part = input_path_param.get("machine_part")
    annotation_json_path = input_path_param.get("annotation_json_path")
    database_path = input_path_param.get("database_path")

    if not machine_part or not annotation_json_path or not database_path:
        raise ValueError("Required input paths or machine part are missing.")

    to_58_included = preprocessing_param.get("to_58_included", False)
    preprocessor = LSTMPreprocessor(database_path, machine_part, annotation_json_path)

    if machine_part == "De-Clamping":
        to_58_included = True

    sensors_df, target_df = preprocessor.read_data(machine_part)

    # Drop irrelevant sensors to reduce dimensionality
    eliminated_columns = [
        "PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]",
        "COLLET_ROTATING_Movement_[mm]",
        "BEND-DIE_VERTICAL_Movement_[mm]",
        "PRESSURE-DIE_LATERAL_Movement_[mm]",
    ]
    sensors_df.drop(columns=eliminated_columns, inplace=True)

    # Normalize and resample data per experiment
    target_df = target_df.reset_index(drop=True)
    out = []
    for exp_id, g in target_df.groupby("Experiment_ID"):
        g_clean = g.drop(columns=["Experiment_ID"])
        r = _normalize_experiment(g_clean, n=46)
        r["Experiment_ID"] = exp_id
        out.append(r)
    target_df = pd.concat(out, ignore_index=True)

    sensors_df = sensors_df.reset_index()
    out = []
    for exp_id, g in sensors_df.groupby("Experiment_ID"):
        r = resample_experiment_ultrafast(
            g,
            n=preprocessing_param.get("window_num", 40),
            metric=preprocessing_param.get("agg_metric", "mean"),
        )
        r["Experiment_ID"] = exp_id
        out.append(r)
    sensors_df = pd.concat(out, ignore_index=True)

    # Select target feature columns
    columns = list(target_df.columns[1:])
    feature_idx_start, feature_idx_end = preprocessing_param.get("feature_indices")
    target_feature_names = columns[feature_idx_start - 1 : feature_idx_end - 1]

    if to_58_included:
        sensors_df = sensors_df[sensors_df["Experiment_ID"] >= 58]
        target_df = target_df[target_df["Experiment_ID"] >= 58]

    # Convert grouped data to 3D tensors
    X_train_numpy = preprocessor.group_and_pad(sensors_df, group_col="Experiment_ID")
    Y_train_numpy = preprocessor.group_and_pad(target_df, group_col="Experiment_ID")[
        :, :, feature_idx_start:feature_idx_end
    ]

    X = torch.from_numpy(X_train_numpy).float()
    Y = torch.from_numpy(Y_train_numpy).float()
    sensor_names = list(sensors_df.columns[:-1])

    # Normalize annotation indices to current sequence length
    annot_timesteps, mandrel_extraction_annot_timesteps = _scale_annotation_timesteps(
        annot_timesteps=preprocessing_param.get("annot_timesteps", []),
        mandrel_extraction_timesteps=preprocessing_param.get(
            "mandrel_extraction_annot_timesteps", []
        ),
        sequence_length=X.shape[1],
    )

    return (
        X,
        Y,
        sensor_names,
        target_feature_names,
        annot_timesteps,
        mandrel_extraction_annot_timesteps,
    )


def create_data_loaders(X_train: torch.Tensor, Y_train: torch.Tensor,
                        X_val: torch.Tensor, Y_val: torch.Tensor,
                        batch_size: int) -> tuple:
    """
    Create PyTorch DataLoaders for training, validation, and plotting.

    Args:
        X_train: Training input tensor.
        Y_train: Training target tensor.
        X_val: Validation input tensor.
        Y_val: Validation target tensor.
        batch_size: Batch size for training loader.

    Returns:
        Tuple of (train_loader, val_loader, plot_loader)
    """
    train_ds = ProcessDataset(X_train, Y_train)
    val_ds = ProcessDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    plot_loader = DataLoader(val_ds, batch_size=min(64, len(val_ds)), shuffle=False)

    return train_loader, val_loader, plot_loader
