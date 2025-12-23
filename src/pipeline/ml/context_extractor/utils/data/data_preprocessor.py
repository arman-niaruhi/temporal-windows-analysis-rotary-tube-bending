import torch
from torch.utils.data import Dataset
from src.pipeline.ml.context_extractor.utils.data.data_resampler import (
    resample_experiment_ultrafast,
)
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from src.pipeline.preprocessing.loader import DataLoader as DataLoaderETL
import json



class LSTMPreprocessor:
    def __init__(self, database_path, machine_part, annotation_json_path):
        self.machine_par = machine_part
        self.sensor_df = None
        self.target_df = None
        self._feature_cols = None
        self.annotation_json_path = annotation_json_path
        self.database_path = database_path

    def read_data(self, label_name):
        loader = DataLoaderETL(self.database_path)
        dataframes = loader.load_all_data_from_sqlite()
        sensors_df = dataframes['machine_and_movement']

        with open(self.annotation_json_path, "r") as f:
            labels = json.load(f)

        if label_name == "All":
            sensors_df.index = sensors_df["Time_[s]"]
            sensors_df.drop(columns=["Time_[s]"], inplace=True)
            return sensors_df, dataframes['arc']
        
        label_windows = (
            pd.DataFrame([
                {
                    "Experiment_ID": int(exp_id),
                    "start": phase["start"],
                    "end": phase["end"]
                }
                for exp_id, phases in labels.items()
                for phase in phases
                if phase["label"] == label_name
            ])
        )
        df_label = (
                sensors_df
                .merge(label_windows, on="Experiment_ID", how="inner")
                .query("`Time_[s]` >= start and `Time_[s]` <= end")
                .drop(columns=["start", "end"])
            )
        df_label.index = df_label["Time_[s]"]
        df_label.drop(columns=["Time_[s]"], inplace=True)
        return df_label, dataframes['arc']

    def feature_selection(self, variance_threshold=0):
        """Select features based on variance threshold."""
        if self.sensor_df is None:
            raise ValueError("Data not loaded. Run read_data() first.")
        self.sensor_df = self.sensor_df.loc[:, self.sensor_df.var() > variance_threshold]
        return self.sensor_df

    def get_feature_cols(self):
        return self._feature_cols
    
    def get_dfs(self):
        """Get the sensor and target dataframes"""
        if self.sensor_df is None and self.target_df is None:
            print("DataFrames not assigned!")
            return
        return self.sensor_df, self.target_df

    def group_and_pad(self, df, group_col="Experiment_ID", exclude_cols=None):
        """
        Groups a DataFrame by a column, pads with NaN to match the longest group, 
        and returns a 3D numpy array with NaN replaced by 0.
        """
        if exclude_cols is None:
            exclude_cols = [group_col]
        groups_list = []
        max_len = df.groupby(group_col).size().max()
        num_features = df.shape[1] - len(exclude_cols)

        for _, group in df.groupby(group_col):
            values = group.drop(columns=exclude_cols).values
            padded = np.full((max_len, num_features), np.nan)
            padded[:values.shape[0], :] = values
            groups_list.append(padded)

        array_3d = np.array(groups_list, dtype=float)
        return np.nan_to_num(array_3d, nan=0.0)

    def normalize_column(self, column_name):
        """Normalize arbitrary column in target_df"""
        if self.target_df is None or column_name not in self.target_df.columns:
            print(f"Warning: target_df is None or column '{column_name}' does not exist.")
            return
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_df[column_name] = scaler.fit_transform(self.target_df[[column_name]])
        return self.target_df
    
    def normalize_angle(self, col="Angle[degree]ORDistance[mm]"):
        """Normalize a specific target column for window algorithm."""
        return self.normalize_column(col)
    
    def prepare_rf_data(self, X, Y):
        """
        Prepares data for RandomForestRegressor:
        Flattens sequences and appends normalized angle as extra feature.
        Returns X_rf, Y_rf as 2D arrays.
        """
        X_rf = []
        Y_rf = []

        num_samples, seq_len, num_features = X.shape
        num_angles = Y.shape[1]

        for sample_idx in range(num_samples):
            for angle_idx in range(num_angles):
                x_seq = X[sample_idx].flatten()
                degree = angle_idx / (num_angles - 1)  # normalized angle
                x_with_angle = np.append(x_seq, degree)
                X_rf.append(x_with_angle)
                Y_rf.append(Y[sample_idx, angle_idx])

        X_rf = np.array(X_rf)
        Y_rf = np.array(Y_rf)
        return X_rf, Y_rf


class ProcessDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X.float()
        self.Y = Y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def _normalize_experiment(group, n=46):
    if len(group) > n:
        # Just take the first 46 rows
        return group.iloc[:n].copy()
    else:
        # Already 46 rows
        return group.copy()


def prepare_data(input_path_param, preprocessing_param):
    machine_part = input_path_param.get("machine_part", "")
    annotation_json_path = input_path_param.get("annotation_json_path")
    database_path = input_path_param.get("database_path")
    preprocessor = LSTMPreprocessor(
        database_path,
        machine_part,
        annotation_json_path
    )
    to_58_included = preprocessing_param.get("to_58_included", False)
    sensors_df, target_df = preprocessor.read_data(machine_part)
    ELIMINATED_COLUMNS = [
                        "PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]", 
                        "COLLET_ROTATING_Movement_[mm]", 
                        "BEND-DIE_VERTICAL_Movement_[mm]", 
                        "PRESSURE-DIE_LATERAL_Movement_[mm]"
                        ]
    sensors_df.drop(columns=ELIMINATED_COLUMNS, inplace=True)
    if input_path_param:
        target_df = target_df.reset_index(drop=True)

        out = []
        for exp_id, g in target_df.groupby("Experiment_ID"):
            # Drop grouping column before processing
            g_clean = g.drop(columns=["Experiment_ID"])

            r = _normalize_experiment(g_clean, n=46)

            # Reattach group ID after processing
            r["Experiment_ID"] = exp_id

            out.append(r)

        import pandas as pd

        target_df = pd.concat(out, ignore_index=True)

        sensors_df = sensors_df.reset_index()
        out = []
        for exp_id, g in sensors_df.groupby("Experiment_ID"):
            r = resample_experiment_ultrafast(
                g,
                n=preprocessing_param.get("window_num", 40),
                metric=preprocessing_param.get("agg_mertic", "mean"),
            )
            r["Experiment_ID"] = exp_id
            out.append(r)
        import pandas as pd

        sensors_df = pd.concat(out, ignore_index=True)

    columns = list(target_df.columns[1:])
    feature_idx_start, feature_idx_end = preprocessing_param.get("feature_indices")
    target_feature_names = columns[feature_idx_start-1:feature_idx_end-1]

    if machine_part == "DECLAMPING":
        to_58_included = True

    if to_58_included:
        # Subset the dataframes
        sensors_df = sensors_df[sensors_df["Experiment_ID"] >= 58]
        target_df = target_df[target_df["Experiment_ID"] >= 58]

    X_train_numpy = preprocessor.group_and_pad(sensors_df, group_col="Experiment_ID")
    Y_train_numpy = preprocessor.group_and_pad(target_df, group_col="Experiment_ID")[
        :, :, feature_idx_start:feature_idx_end
    ]

    X = torch.from_numpy(X_train_numpy).float()
    Y = torch.from_numpy(Y_train_numpy).float()
    sensor_names = list(sensors_df.columns[:-1])
    annot_timesteps = preprocessing_param.get("annot_timesteps", None)
    mandrel_extraction_annot_timesteps = preprocessing_param.get("mandrel_extraction_annot_timesteps", None)
    N = X.shape[1]
    annot_timesteps = [int((idx / 1743) * N) for idx in annot_timesteps]
    mandrel_extraction_annot_timesteps = [int((idx / 1743) * N) for idx in mandrel_extraction_annot_timesteps]
    return X, Y, sensor_names, target_feature_names, annot_timesteps, mandrel_extraction_annot_timesteps
