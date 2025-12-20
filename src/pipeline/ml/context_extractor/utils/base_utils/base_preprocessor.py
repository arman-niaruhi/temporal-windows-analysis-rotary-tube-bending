import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.pipeline.preprocessing.loader import DataLoader as DataLoaderETL
import json


import sys
import os
current_dir = os.getcwd()
project_root = os.path.join(current_dir)
project_root = os.path.abspath(project_root)
sys.path.insert(0, project_root)
DATABASE_PATH = f"{project_root}/data/processed/tube_geometry.db"
class BasePreprocessor:
    def __init__(self, machine_part, annotation_json_path):
        self.machine_par = machine_part
        self.sensor_df = None
        self.target_df = None
        self._feature_cols = None
        self.annotation_json_path = annotation_json_path

    def read_data(self, label_name):
        loader = DataLoaderETL(DATABASE_PATH)
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



