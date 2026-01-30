import logging
from typing import Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader

from src.pipeline.ml.context_extractor.utils.data.data_resampler import (
    resample_experiment_ultrafast,
)
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.pipeline.preprocessing.loader import DataLoader as DataLoaderETL
import json

# Initialize logger for monitoring execution
logger = logging.getLogger(__name__)

def _fit_scaler(
    df: pd.DataFrame,
    cols: list[str],
    scaler_type: str = "minmax",
) -> tuple[pd.DataFrame, MinMaxScaler | StandardScaler | None]:
    if not cols:
        return df, None
    scaler_type = scaler_type.lower()
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    values = df[cols].to_numpy(dtype="float32")
    scaler.fit(values)
    df_scaled = df.copy()
    df_scaled[cols] = scaler.transform(values)
    return df_scaled, scaler


def _apply_scaler(
    df: pd.DataFrame,
    cols: list[str],
    scaler: MinMaxScaler | StandardScaler | None,
) -> pd.DataFrame:
    if scaler is None or not cols:
        return df
    df_scaled = df.copy()
    df_scaled[cols] = scaler.transform(df[cols].to_numpy(dtype="float32"))
    return df_scaled


class LSTMPreprocessor:
    """
    Preprocessor class for LSTM-based time-series modeling of machine sensor data.

    This class handles:
    - Reading raw sensor and target data
    - Feature selection based on variance thresholds
    - Normalization of sensor and target data
    - Grouping and padding of sequences to prepare 3D tensors
    """

    def __init__(self, database_path: str, process_part: str, annotation_json_path: str) -> None:
        """
        Initialize the preprocessor.

        Args:
            database_path: Path to the SQLite database containing raw data.
            process_part: Identifier of the machine part to model (e.g., 'De-Clamping').
            annotation_json_path: Path to JSON file containing label annotations.
        """
        self.process_part = process_part
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

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, Springback: torch.Tensor, experiment_configurations: torch.Tensor) -> None:
        self.X = X.float()
        self.Y = Y.float()
        self.Springback = Springback.float().squeeze()
        self.experiment_configurations = experiment_configurations

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Springback[idx], self.experiment_configurations[idx]


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


def _to_tensor_split(
    preprocessor,
    sensors_df,
    target_df,
    feature_idx_start,
    feature_idx_end,
):
    """
    Convert grouped DataFrames into 3D tensors for LSTM input/target.
    """
    X_np = preprocessor.group_and_pad(sensors_df, group_col="Experiment_ID")
    Y_np = preprocessor.group_and_pad(target_df, group_col="Experiment_ID")[
        :, :, feature_idx_start:feature_idx_end
    ]
    return torch.from_numpy(X_np).float(), torch.from_numpy(Y_np).float()


def prepare_data(input_path_param: dict, preprocessing_param: dict) -> Any:
    """
    Data preparation for LSTM training, returning train and validation sets.

    Returns:
        X_train, Y_train: training tensors
        X_val, Y_val: validation tensors
        sensor_names: feature names
        target_feature_names: target columns
        annot_timesteps: scaled annotation indices
        mandrel_extraction_annot_timesteps: scaled mandrel extraction indices
        normalization_info: scalers used for normalization (if enabled)
    """
    process_part = input_path_param.get("process_part")
    annotation_json_path = input_path_param.get("annotation_json_path")
    database_path = input_path_param.get("database_path")

    if not process_part or not annotation_json_path or not database_path:
        raise ValueError("Required input paths or machine part are missing.")

    normalize = preprocessing_param.get("normalize", False)
    scaler_type = preprocessing_param.get("scaler", "minmax")
    preprocessor = LSTMPreprocessor(database_path, process_part, annotation_json_path)

    sensors_df, target_df = preprocessor.read_data(process_part)

    # Drop irrelevant sensors
    eliminated_columns = [
        "PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]",
        "COLLET_ROTATING_Movement_[mm]",
        "BEND-DIE_VERTICAL_Movement_[mm]",
        "PRESSURE-DIE_LATERAL_Movement_[mm]",
        "MACHINE_PRESSURE-DIE_AXIAL_Max_Torque_[%]",
        "PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]",
        #"MACHINE_BEND_DIE_LATERAL_Max_Torque_[%]",
        #"MACHINE_BEND_DIE_ROTATING_Max_Torque_[%]",
        #"MACHINE_BEND_DIE_VERTICAL_Max_Torque_[%]",
        #"MACHINE_CLAMP_DIE_LATERAL_Max_Torque_[%]",
        #"MACHINE_COLLET_AXIAL_Max_Torque_[%]",
        #"MACHINE_MANDREL_AXIAL_Max_Torque_[%]",
        #"MACHINE_PRESSURE_DIE_LATERAL_Max_Torque_[%]",
        #"BEND-DIE_LATERAL_Movement_[mm]",
        #"BEND-DIE_ROTATING_Angle_[°]",
        #"CLAMP-DIE_LATERAL_Movement_[mm]",
        #"COLLET_AXIAL_Movement_[mm]",
        #"MANDREL_AXIAL_Movement_[mm]",
        #"PRESSURE-DIE_AXIAL_Movement_[mm]",
        
    ]

    sensors_df.drop(columns=eliminated_columns, inplace=True)
  
    # Resample and normalize target and sensor data
    target_df = target_df.reset_index(drop=True)
    crosscuts_list, spring_backs_list = [], []
    for exp_id, g in target_df.groupby("Experiment_ID"):
        g_clean = g.drop(columns=["Experiment_ID"])
        crosscuts = g_clean.iloc[:45,:].copy()
        spring_back = g_clean.iloc[-1:,].copy()
        spring_back['Angle[degree]ORDistance[mm]'] = 46.0 - spring_back['Angle[degree]ORDistance[mm]']
        
        crosscuts["Experiment_ID"] = exp_id
        spring_back["Experiment_ID"] = exp_id
        crosscuts_list.append(crosscuts)
        spring_backs_list.append(spring_back)
    target_df = pd.concat(crosscuts_list, ignore_index=True)
    spring_backs_df = pd.concat(spring_backs_list, ignore_index=True)

    sensors_df = sensors_df.reset_index()
    
    do_resample = preprocessing_param.get("resample", False)
    if do_resample:
        resampled_sensors = []
        for exp_id, g in sensors_df.groupby("Experiment_ID", sort=False):
            r = resample_experiment_ultrafast(
                g,
                n=preprocessing_param.get("window_num", 40),
                metric=preprocessing_param.get("agg_metric", "mean"),
            )
            r["Experiment_ID"] = exp_id
            resampled_sensors.append(r)

        sensors_df = pd.concat(resampled_sensors, ignore_index=True)
    else:
    
        # Keep raw sequence ordering when resampling is disabled.
        if "Time_[s]" in sensors_df.columns:
            sensors_df = sensors_df.sort_values(["Experiment_ID", "Time_[s]"])
        else:
            sensors_df = sensors_df.sort_values(["Experiment_ID"])

    sensors_df = sensors_df.drop(columns=["Time_[s]"], errors="ignore")
    # Select target features
    columns = target_df.columns.tolist()
    feature_idx_start, feature_idx_end = preprocessing_param.get("feature_indices")

    target_feature_names = columns[feature_idx_start: feature_idx_end]
    springback_feature_names = columns[:1]
    '''
    if to_58_included:
        sensors_df = sensors_df[sensors_df["Experiment_ID"] >= 58]
        target_df = target_df[target_df["Experiment_ID"] >= 58]
        spring_backs_df = spring_backs_df[spring_backs_df["Experiment_ID"] >= 58]
    '''
    normalization_info = {
        "enabled": normalize,
        "scaler_type": scaler_type,
        "sensor_scaler": None,
        "target_scaler": None,
        "springback_scaler": None,
        "config_scaler": None,
    }

    if normalize:
        sensor_cols = [c for c in sensors_df.columns if c != "Experiment_ID"]
        sensors_df, sensor_scaler = _fit_scaler(sensors_df, sensor_cols, scaler_type)
        normalization_info["sensor_scaler"] = sensor_scaler

        target_df, target_scaler = _fit_scaler(target_df, target_feature_names, scaler_type)
        normalization_info["target_scaler"] = target_scaler

        spring_backs_df, springback_scaler = _fit_scaler(
            spring_backs_df, springback_feature_names, scaler_type
        )
        normalization_info["springback_scaler"] = springback_scaler

    # --------------------------------------------------
    # Train / Validation Split
    # --------------------------------------------------
    # Read groups from the configured JSON (fallback to legacy default)
    split_config_path = preprocessing_param.get(
        "split_config_path",
        "config/data-split-config/train_test_split.json",
    )
    with open(split_config_path, "r") as f:
        experiment_groups = json.load(f)

    if experiment_groups is None:
        raise ValueError("experiment_groups must be provided for train/val split")

    train_groups = experiment_groups['train_groups']
    train_groups = [item for sublist in train_groups for item in sublist]
     
    test_groups = experiment_groups['test_groups']
    test_groups = [item for sublist in test_groups for item in sublist]

    train_springbacks = spring_backs_df[spring_backs_df["Experiment_ID"].isin(train_groups)]
    test_springbacks= spring_backs_df[spring_backs_df["Experiment_ID"].isin(test_groups)]
    train_targets = target_df[target_df["Experiment_ID"].isin(train_groups)]
    test_targets = target_df[target_df["Experiment_ID"].isin(test_groups)]
    train_sensors = sensors_df[sensors_df["Experiment_ID"].isin(train_groups)]
    testsensors = sensors_df[sensors_df["Experiment_ID"].isin(test_groups)]

    # --------------------------------------------------
    # Experiment configuration alignment + normalization
    # --------------------------------------------------
    experiment_configurations = pd.read_csv("config/data-split-config/experiment_setups.csv").reset_index(drop=True)
    feature_cols = experiment_configurations.columns.drop("Experiment_ID")
    if normalize:
        experiment_configurations, config_scaler = _fit_scaler(
            experiment_configurations, list(feature_cols), scaler_type
        )
        normalization_info["config_scaler"] = config_scaler

    train_exp_ids = sorted(train_sensors["Experiment_ID"].unique())
    test_exp_ids = sorted(testsensors["Experiment_ID"].unique())

    config_by_id = experiment_configurations.set_index("Experiment_ID")

    missing_train = [eid for eid in train_exp_ids if eid not in config_by_id.index]
    missing_test = [eid for eid in test_exp_ids if eid not in config_by_id.index]
    if missing_train or missing_test:
        raise ValueError(
            f"Missing experiment config for IDs. "
            f"train_missing={missing_train[:10]} test_missing={missing_test[:10]}"
        )

    experiment_configurations_train = config_by_id.loc[train_exp_ids, feature_cols]
    experiment_configurations_test = config_by_id.loc[test_exp_ids, feature_cols]

    experiment_configurations_train = torch.from_numpy(
        experiment_configurations_train.to_numpy(dtype="float32")
    )
    experiment_configurations_test = torch.from_numpy(
        experiment_configurations_test.to_numpy(dtype="float32")
    )

    # Convert to tensors for targets
    X_train, Y_train = _to_tensor_split(
        preprocessor, train_sensors, train_targets, feature_idx_start, feature_idx_end
    )
    X_test, Y_test = _to_tensor_split(
        preprocessor, testsensors, test_targets, feature_idx_start, feature_idx_end
    )
    
    # Convert to tensors for springbacks
    _, springbacks_train = _to_tensor_split(
        preprocessor, train_sensors, train_springbacks, 0, 1
    )
    _, springbacks_test = _to_tensor_split(
        preprocessor, testsensors, test_springbacks, 0, 1
    )

    sensor_names = [c for c in sensors_df.columns if c != "Experiment_ID"]

    # Scale annotation indices based on training sequence length
    annot_timesteps, mandrel_extraction_annot_timesteps = _scale_annotation_timesteps(
        annot_timesteps=preprocessing_param.get("annot_timesteps", []),
        mandrel_extraction_timesteps=preprocessing_param.get(
            "mandrel_extraction_annot_timesteps", []
        ),
        sequence_length=X_train.shape[1],
    )

    return (
        X_train,
        Y_train,
        X_test,
        Y_test,
        springbacks_train,
        springbacks_test,
        experiment_configurations_train,
        experiment_configurations_test,
        sensor_names,
        target_feature_names,
        annot_timesteps,
        mandrel_extraction_annot_timesteps,
        normalization_info,
    )



def create_data_loaders(X_train: torch.Tensor, Y_train: torch.Tensor,
                        X_val: torch.Tensor, Y_val: torch.Tensor,
                        springbacks_train: torch.Tensor, springbacks_val: torch.Tensor,
                        experiment_configurations_train: torch.Tensor, 
                        experiment_configurations_test: torch.Tensor,
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
    train_ds = ProcessDataset(X_train, Y_train, springbacks_train, experiment_configurations_train)
    val_ds = ProcessDataset(X_val, Y_val, springbacks_val, experiment_configurations_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    plot_loader = DataLoader(val_ds, batch_size=min(64, len(val_ds)), shuffle=False)

    return train_loader, val_loader, plot_loader
