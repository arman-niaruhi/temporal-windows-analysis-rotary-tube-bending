import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

logger = logging.getLogger(__name__)


class ClassifierPreprocessor:
    """Preprocessor for classification tasks on sensor data with annotations."""

    def __init__(
        self,
        sensors_df: pd.DataFrame,
        annotation_json: str = "../data/machine-and-movement.json",
    ):
        """
        Initialize preprocessor with sensor data and annotation path.

        Args:
            sensors_df: DataFrame with sensor data
            annotation_json: Path to JSON file with annotation intervals
        """
        logger.info("Initializing ClassifierPreprocessor...")
        self.annotation_json = annotation_json
        self.sensors_df = sensors_df
        self.annotation_dict: Optional[Dict] = None
        self._feature_cols: Optional[pd.Index] = None
        self.label_binarizer: Optional[MultiLabelBinarizer] = None

    def get_feature_cols(self) -> Optional[pd.Index]:
        """Get the feature column names."""
        return self._feature_cols

    def read_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Read sensor data and annotation JSON.

        Returns:
            Tuple of (sensors_df, annotation_dict)
        """
        if "Time_[s]" in self.sensors_df.columns:
            self.sensors_df.index = self.sensors_df["Time_[s]"]
            self.sensors_df = self.sensors_df.drop(columns=["Time_[s]"])

        with open(self.annotation_json, "r") as f:
            self.annotation_dict = json.load(f)

        logger.info("Sensor data and annotations loaded.")
        return self.sensors_df, self.annotation_dict

    def feature_selection(self, variance_threshold: float = 0) -> pd.DataFrame:
        """
        Select features based on variance threshold.

        Args:
            variance_threshold: Minimum variance required to keep a feature

        Returns:
            DataFrame with selected features
        """
        if self.sensors_df is None:
            raise ValueError("Data not loaded. Run read_data() first.")

        logger.info(
            f"Performing feature selection with variance threshold: {variance_threshold}"
        )
        self.sensors_df = self.sensors_df.loc[
            :, self.sensors_df.var() > variance_threshold
        ]
        return self.sensors_df

    def delete_columns(self, eliminated_columns: List[str]) -> pd.DataFrame:
        """
        Delete specified columns from the sensor DataFrame.

        Args:
            eliminated_columns: List of column names to delete

        Returns:
            DataFrame with specified columns removed
        """
        logger.info(f"Deleting columns: {eliminated_columns}")
        self.sensors_df = self.sensors_df.drop(columns=eliminated_columns)
        return self.sensors_df

    def _get_active_labels(self, row: pd.Series) -> List[str]:
        """
        Get all active labels for a given row based on timestamp and experiment ID.

        Args:
            row: DataFrame row with Experiment_ID and timestamp as index

        Returns:
            List of active label strings
        """
        exp_id = row["Experiment_ID"]
        if exp_id not in self.annotation_dict:
            return []

        active_labels = [
            interval.get("label", "No Label")
            for interval in self.annotation_dict[exp_id]
            if float(interval["start"]) <= row.name <= float(interval["end"])
        ]

        return [lbl for lbl in active_labels if lbl]

    def assign_labels(self) -> pd.DataFrame:
        """
        Assign labels from intervals JSON to each timestamp.
        Priority logic: If both 'Bending' and 'Mandrel Extraction' are active,
        assign 'Mandrel Extraction'. Otherwise, assign the first active label.

        Returns:
            DataFrame with assigned labels in 'Label' column
        """
        self._validate_data_loaded()
        self._prepare_dataframe()

        def get_label(row: pd.Series) -> str:
            active_labels = self._get_active_labels(row)

            if not active_labels:
                return "No Label"

            if "Bending" in active_labels and "Mandrel Extraction" in active_labels:
                return "Mandrel Extraction"

            return active_labels[0]

        self.sensors_df["Label"] = self.sensors_df.apply(get_label, axis=1)
        logger.info("Labels assigned to sensor data.")
        return self.sensors_df

    def assign_one_label(self, target_label: Optional[str] = None) -> pd.DataFrame:
        """
        Assign labels from intervals JSON to each timestamp.
        Optionally filter to keep only one target label; all others become 'No Label'.

        Args:
            target_label: Specific label to keep; others become 'No Label'

        Returns:
            DataFrame with assigned labels in 'Label' column
        """
        self._validate_data_loaded()
        self._prepare_dataframe()

        def get_label(row: pd.Series) -> str:
            active_labels = self._get_active_labels(row)

            if not active_labels:
                return "No Label"

            if target_label is not None:
                return target_label if target_label in active_labels else "No Label"

            return active_labels[0]

        self.sensors_df["Label"] = self.sensors_df.apply(get_label, axis=1)
        logger.info("Labels assigned to sensor data.")
        return self.sensors_df

    def keep_only_one_active_label(self) -> pd.DataFrame:
        """
        Automatically detect the dominant non-'No Label' class in the dataset
        and replace all other labels with 'No Label'.

        Returns:
            DataFrame with only one active label retained
        """
        if self.sensors_df is None or "Label" not in self.sensors_df.columns:
            raise ValueError("Data not labeled yet. Run assign_labels() first.")

        self.sensors_df = self.sensors_df.copy()

        unique_labels = [
            lbl for lbl in self.sensors_df["Label"].unique() if lbl != "No Label"
        ]

        if not unique_labels:
            raise ValueError(
                "No valid labels found — dataset only contains 'No Label'."
            )

        label_counts = self.sensors_df["Label"].value_counts()
        target_label = label_counts[label_counts.index != "No Label"].idxmax()

        logger.info(f"Keeping only label: {target_label}")

        self.sensors_df["Label"] = self.sensors_df["Label"].apply(
            lambda lbl: lbl if lbl == target_label else "No Label"
        )

        return self.sensors_df

    def normalize_and_encode_labels(self) -> pd.DataFrame:
        """
        Normalize features, convert to numeric, fill NaNs, and encode labels.

        Returns:
            DataFrame with normalized features and encoded labels
        """
        if self.sensors_df is None:
            raise ValueError("Data not loaded. Run read_data() first.")

        self._feature_cols = self.sensors_df.columns.difference(
            ["Experiment_ID", "Label"]
        )

        self.sensors_df[self._feature_cols] = self.sensors_df[self._feature_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        self.sensors_df[self._feature_cols] = self.sensors_df[
            self._feature_cols
        ].fillna(0.0)

        self.sensors_df["Label_encoded"] = LabelEncoder().fit_transform(
            self.sensors_df["Label"]
        )

        logger.info("Features normalized and labels encoded.")
        return self.sensors_df

    def split_experiments(
        self,
        experiment_groups,
        test_ratio: float = 0.1,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[int]]]:
        """
        Split data by experiment groups into train, validation, and test sets.
        Ensures that experiment IDs from the same group are never split.

        Args:
            experiment_groups: List of lists, where each sublist contains experiment IDs
                             that belong to the same group
            test_ratio: Proportion of data to allocate to test set
            val_ratio: Proportion of data to allocate to validation set
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_df, val_df, test_df, splits_dict)
        """
        if self.sensors_df is None:
            raise ValueError("Data not loaded. Run read_data() first.")

        sensor_df = self.sensors_df.copy()
        sensor_df["Experiment_ID"] = sensor_df["Experiment_ID"].astype(int)

        np.random.seed(seed)
        shuffled_groups = experiment_groups.copy()
        np.random.shuffle(shuffled_groups)

        n_total_groups = len(shuffled_groups)
        n_test_groups = int(test_ratio * n_total_groups)
        n_val_groups = int(val_ratio * n_total_groups)

        test_groups = shuffled_groups[:n_test_groups]
        val_groups = shuffled_groups[n_test_groups : n_test_groups + n_val_groups]
        train_groups = shuffled_groups[n_test_groups + n_val_groups :]

        train_exps = [eid for group in train_groups for eid in group]
        val_exps = [eid for group in val_groups for eid in group]
        test_exps = [eid for group in test_groups for eid in group]

        train_df = sensor_df[sensor_df["Experiment_ID"].isin(train_exps)].copy()
        val_df = sensor_df[sensor_df["Experiment_ID"].isin(val_exps)].copy()
        test_df = sensor_df[sensor_df["Experiment_ID"].isin(test_exps)].copy()

        logger.info(
            f"Data split: train ({len(train_exps)} exp), "
            f"val ({len(val_exps)} exp), test ({len(test_exps)} exp)"
        )

        return (
            train_df,
            val_df,
            test_df,
            {"train": train_exps, "val": val_exps, "test": test_exps},
        )

    def create_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        DatasetClass: type,
    ) -> Tuple:
        """
        Create ML datasets from DataFrames using a given dataset class.

        Args:
            train_df: DataFrame for training set
            val_df: DataFrame for validation set
            test_df: DataFrame for test set
            DatasetClass: Class to instantiate datasets (should accept DataFrame
                         and feature columns)

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self._feature_cols is None:
            raise ValueError(
                "Feature columns not defined. Run normalize_and_encode_labels() first."
            )

        train_dataset = DatasetClass(train_df, self._feature_cols)
        val_dataset = DatasetClass(val_df, self._feature_cols)
        test_dataset = DatasetClass(test_df, self._feature_cols)

        logger.info("Datasets created successfully.")
        return train_dataset, val_dataset, test_dataset

    def assign_multi_labels(self) -> pd.DataFrame:
        """
        Assign all active labels from the JSON intervals to each timestamp.
        Each row may contain zero, one, or multiple labels.

        Returns:
            DataFrame with assigned labels in 'Labels' column (list of labels)
        """
        self._validate_data_loaded()
        self._prepare_dataframe()

        def get_all_labels(row: pd.Series) -> List[str]:
            active_labels = self._get_active_labels(row)
            return [lbl for lbl in active_labels if lbl != "No Label"]

        self.sensors_df["Labels"] = self.sensors_df.apply(get_all_labels, axis=1)
        logger.info("Multi-labels assigned to sensor data.")
        return self.sensors_df

    def encode_multi_labels(self) -> pd.DataFrame:
        """
        Convert 'Labels' list column into multi-hot encoded columns.
        Produces columns: 'Label_Bending', 'Label_Mandrel Extraction', etc.

        Returns:
            DataFrame with multi-hot encoded label columns
        """
        if self.sensors_df is None or "Labels" not in self.sensors_df.columns:
            raise ValueError("Run assign_multi_labels() first.")

        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(self.sensors_df["Labels"])

        for label, col in zip(mlb.classes_, encoded.T):
            self.sensors_df[f"Label_{label}"] = col

        self.label_binarizer = mlb

        logger.info(f"Multi-labels encoded into {len(mlb.classes_)} columns.")
        return self.sensors_df

    def _validate_data_loaded(self):
        """Validate that data and annotations are loaded."""
        if self.sensors_df is None or self.annotation_dict is None:
            raise ValueError("Data not loaded. Run read_data() first.")

    def _prepare_dataframe(self):
        """Prepare DataFrame for label assignment."""
        self.sensors_df = self.sensors_df.copy()
        self.sensors_df.index = self.sensors_df.index.astype(float)
        self.sensors_df["Experiment_ID"] = self.sensors_df["Experiment_ID"].astype(str)
