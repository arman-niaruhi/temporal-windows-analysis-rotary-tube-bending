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
    
    def assign_one_label(self, target_label=None):
        """
        Assign labels from intervals JSON to each timestamp.
        Optionally assign only one target label; all others become 'No Label'.
        """
        if self.sensor_df is None or self.annotation_dict is None:
            raise ValueError("Data not loaded. Run read_data() first.")

        self.sensor_df = self.sensor_df.copy()
        self.sensor_df.index = self.sensor_df.index.astype(float)
        self.sensor_df["Experiment_ID"] = self.sensor_df["Experiment_ID"].astype(str)

        def get_label(row):
            exp_id = row["Experiment_ID"]
            if exp_id not in self.annotation_dict:
                return "No Label"

            # Find all active labels at this timestamp
            active_labels = [
                interval.get("label", "No Label")
                for interval in self.annotation_dict[exp_id]
                if float(interval["start"]) <= row.name <= float(interval["end"])
            ]

            if not active_labels:
                return "No Label"

            # If user wants only one specific label, apply filtering
            if target_label is not None:
                return target_label if target_label in active_labels else "No Label"

            # Otherwise return the first found label (default behavior)
            return active_labels[0]

        self.sensor_df["Label"] = self.sensor_df.apply(get_label, axis=1)
        return self.sensor_df

    def keep_only_one_active_label(self):
        """
        Automatically detect the single non-'No Label' class in the dataset
        and replace all other labels with 'No Label'.

        Example:
            If labels are ['Bending', 'Mandrel Extraction', 'No Label'],
            it keeps only 'Bending' and converts all others to 'No Label'.
        """
        if self.sensor_df is None or "Label" not in self.sensor_df.columns:
            raise ValueError("Data not labeled yet. Run assign_labels() first.")
        
        self.sensor_df = self.sensor_df.copy()

        # Find all unique labels except "No Label"
        unique_labels = [lbl for lbl in self.sensor_df["Label"].unique() if lbl != "No Label"]

        if not unique_labels:
            raise ValueError("No valid labels found — dataset only contains 'No Label'.")
        
        # Automatically choose the dominant non-"No Label" class
        label_counts = self.sensor_df["Label"].value_counts()
        target_label = label_counts[label_counts.index != "No Label"].idxmax()

        # Replace all other labels with "No Label"
        self.sensor_df["Label"] = self.sensor_df["Label"].apply(
            lambda lbl: lbl if lbl == target_label else "No Label"
        )
        
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

    
    def split_experiments(self, experiment_groups, test_ratio=0.1, val_ratio=0.2, seed=42):
        """
        Split data by experiment groups into train, validation, and test sets,
        ensuring that experiment IDs from the same group are never split.
        
        experiment_groups: list of lists, each containing experiment IDs from your JSON
        """
        if self.sensor_df is None:
            raise ValueError("Data not loaded. Run read_data() first.")
        
        sensor_df = self.sensor_df.copy()
        sensor_df["Experiment_ID"] = sensor_df["Experiment_ID"].astype(int)

        
        # Shuffle experiment groups as units
        np.random.seed(seed)
        np.random.shuffle(experiment_groups)

        n_total_groups = len(experiment_groups)
        n_test_groups = int(test_ratio * n_total_groups)
        n_val_groups = int(val_ratio * n_total_groups)

        # Assign groups to splits
        test_groups = experiment_groups[:n_test_groups]
        val_groups = experiment_groups[n_test_groups:n_test_groups + n_val_groups]
        train_groups = experiment_groups[n_test_groups + n_val_groups:]

        # Flatten experiment IDs for each split
        train_exps = [eid for group in train_groups for eid in group]
        val_exps = [eid for group in val_groups for eid in group]
        test_exps = [eid for group in test_groups for eid in group]

        # Filter DataFrame by experiment IDs
        train_df = sensor_df[sensor_df["Experiment_ID"].isin(train_exps)].copy()
        val_df = sensor_df[sensor_df["Experiment_ID"].isin(val_exps)].copy()
        test_df = sensor_df[sensor_df["Experiment_ID"].isin(test_exps)].copy()
        
        # Print all experiment IDs for each set
        print(f"Train experiment IDs ({len(train_exps)}): {sorted(train_exps)}")
        print(f"Validation experiment IDs ({len(val_exps)}): {sorted(val_exps)}")
        print(f"Test experiment IDs ({len(test_exps)}): {sorted(test_exps)}")

        return train_df, val_df, test_df, {"train": train_exps, "val": val_exps, "test": test_exps}

    
    def create_datasets(self, train_df, val_df, test_df, DatasetClass):
        """Create ML datasets from DataFrames using a given dataset class."""
        if self._feature_cols is None:
            raise ValueError("Feature columns not defined. Run normalize_and_encode_labels() first.")

        train_dataset = DatasetClass(train_df, self._feature_cols)
        val_dataset = DatasetClass(val_df, self._feature_cols)
        test_dataset = DatasetClass(test_df, self._feature_cols)
        return train_dataset, val_dataset, test_dataset
