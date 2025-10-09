import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class ClassifierPreprocessor:
    def __init__(self, sensors_path="../data/features.csv", annotation_json="../data/machine-and-movement.json"):
        self.sensors_path = sensors_path
        self.annotation_json = annotation_json
        self.sensor_df = None
        self.annotation_dict = None
        self._feature_cols = None
    
    def get_feature_cols(self):
        return self._feature_cols

    def read_data(self):
        """Read sensor CSV and annotation JSON."""
        self.sensor_df = pd.read_csv(self.sensors_path, index_col="Time_[s]")
        with open(self.annotation_json, "r") as f:
            self.annotation_dict = json.load(f)
        return self.sensor_df, self.annotation_dict

    def feature_selection(self, variance_threshold=0):
        """Select features based on variance threshold."""
        if self.sensor_df is None:
            raise ValueError("Data not loaded. Run read_data() first.")
        self.sensor_df = self.sensor_df.loc[:, self.sensor_df.var() > variance_threshold]
        return self.sensor_df

    def assign_labels(self):
        """Assign labels from intervals JSON to each timestamp."""
        if self.sensor_df is None or self.annotation_dict is None:
            raise ValueError("Data not loaded. Run read_data() first.")

        self.sensor_df = self.sensor_df.copy()
        self.sensor_df.index = self.sensor_df.index.astype(float)
        self.sensor_df["Experiment_ID"] = self.sensor_df["Experiment_ID"].astype(str)

        def get_label(row):
            exp_id = row["Experiment_ID"]
            if exp_id not in self.annotation_dict:
                return ["No Label"]

            # Find all labels at this timestamp
            active_labels = [
                interval.get("label", "No Label")
                for interval in self.annotation_dict[exp_id]
                if float(interval["start"]) <= row.name <= float(interval["end"])
            ]

            # Only keep row if both 'bending' and 'mandrel_extraction' are present
            if "Bending" in active_labels and "Mandrel Extraction" in active_labels:
                return "Mandrel Extraction"
            elif active_labels:
                return active_labels[0]
            else:
                return "No Label"


        self.sensor_df["Label"] = self.sensor_df.apply(get_label, axis=1)
        return self.sensor_df

    def normalize_and_encode_labels(self):
        """Normalize features, convert to numeric, fill NaNs, and encode labels."""
        if self.sensor_df is None:
            raise ValueError("Data not loaded. Run read_data() first.")

        self._feature_cols = self.sensor_df.columns.difference(['Experiment_ID', 'Label'])

        self.sensor_df[self._feature_cols] = self.sensor_df[self._feature_cols].apply(pd.to_numeric, errors='coerce')
        self.sensor_df[self._feature_cols] = self.sensor_df[self._feature_cols].fillna(0.0)

        self.sensor_df[self._feature_cols] = MinMaxScaler().fit_transform(self.sensor_df[self._feature_cols])
        self.sensor_df['Label_encoded'] = LabelEncoder().fit_transform(self.sensor_df['Label'])
        return self.sensor_df

    def split_experiments(self, test_ratio=0.1, val_ratio=0.2, seed=42):
        """Split data by experiments into train, validation, and test sets."""
        if self.sensor_df is None:
            raise ValueError("Data not loaded. Run read_data() first.")

        sensor_df = self.sensor_df.copy()
        experiments = sensor_df["Experiment_ID"].unique()
        np.random.seed(seed)
        np.random.shuffle(experiments)

        n_total = len(experiments)
        n_test = int(test_ratio * n_total)
        n_val = int(val_ratio * n_total)

        test_exps = experiments[:n_test]
        val_exps = experiments[n_test:n_test + n_val]
        train_exps = experiments[n_test + n_val:]

        train_df = sensor_df[sensor_df["Experiment_ID"].isin(train_exps)].copy()
        val_df = sensor_df[sensor_df["Experiment_ID"].isin(val_exps)].copy()
        test_df = sensor_df[sensor_df["Experiment_ID"].isin(test_exps)].copy()

        print(f"Total experiments: {n_total}")
        print(f"Train experiments: {len(train_exps)}")
        print(f"Val experiments: {len(val_exps)}")
        print(f"Test experiments: {len(test_exps)}")

        experiment_ids = test_exps.tolist()
        return train_df, val_df, test_df, experiment_ids
    
    def create_datasets(self, train_df, val_df, test_df, DatasetClass):
        """Create ML datasets from DataFrames using a given dataset class."""
        if self._feature_cols is None:
            raise ValueError("Feature columns not defined. Run normalize_and_encode_labels() first.")

        train_dataset = DatasetClass(train_df, self._feature_cols)
        val_dataset = DatasetClass(val_df, self._feature_cols)
        test_dataset = DatasetClass(test_df, self._feature_cols)
        return train_dataset, val_dataset, test_dataset
