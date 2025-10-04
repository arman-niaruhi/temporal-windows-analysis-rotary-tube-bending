import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class ClusteringPreprocessor:
    def __init__(self, sensors_path="../data/features.csv", target_path="../data/machine-and-movement.json"):
        self.sensors_path = sensors_path
        self.target_path = target_path
        self.sensor_df = None
        self.target_df = None
        self._feature_cols = None
    
    def get_feature_cols(self):
        return self._feature_cols

    def read_data(self):
        """Read sensor CSV and annotation JSON."""
        self.sensor_df = pd.read_csv(self.sensors_path, index_col="Time_[s]")
        self.target_df = pd.read_csv(self.target_path, index_col=False)
        return self.sensor_df, self.target_df

    def feature_selection(self, variance_threshold=0):
        """Select features based on variance threshold."""
        if self.sensor_df is None:
            raise ValueError("Data not loaded. Run read_data() first.")
        self.sensor_df = self.sensor_df.loc[:, self.sensor_df.var() > variance_threshold]
        return self.sensor_df
    
    def normalize_column(self, column_name):
        """Normalize arbitrary column"""
        if self.target_df is None:
            print("The target df is not assigned!")
            return
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_df[column_name] = scaler.fit_transform(self.target_df[[column_name]])
        return self.target_df
            
    def get_dfs(self):
        """Get the sensor and target dataframes"""
        if self.sensor_df is None and self.target_df is None:
            print("The target df is not assigned!")
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
        