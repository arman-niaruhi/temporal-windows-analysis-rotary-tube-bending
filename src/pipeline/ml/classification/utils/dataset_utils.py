import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class SegmentDataset3DSequence(Dataset):
    """
    Dataset that groups timesteps by Experiment_ID into 3D tensors.
    Each sample is one complete experiment with shape (max_timesteps, num_features).
    Labels are per-timestep: (max_timesteps,)
    """

    def __init__(self, sensor_df, feature_cols):
        self.experiments = []
        self.labels = []
        self.experiment_ids = []

        self.unique_labels = sorted(sensor_df["Label"].unique())
        self.label_to_idx = {l: i for i, l in enumerate(self.unique_labels)}

        self.max_len = sensor_df.groupby("Experiment_ID").size().max()
        self.num_features = len(feature_cols)

        for exp_id, group in sensor_df.groupby("Experiment_ID"):
            values = group[feature_cols].values.astype(np.float32)
            seq_len = values.shape[0]

            padded_features = np.zeros(
                (self.max_len, self.num_features), dtype=np.float32
            )
            padded_features[:seq_len, :] = values

            padded_labels = np.full(self.max_len, -1, dtype=np.int64)
            label_indices = group["Label"].map(self.label_to_idx).values
            padded_labels[:seq_len] = label_indices

            self.experiments.append(torch.tensor(padded_features, dtype=torch.float32))
            self.labels.append(torch.tensor(padded_labels, dtype=torch.long))
            self.experiment_ids.append(exp_id)
        logger.info(
            f"Created SegmentDataset3DSequence with {len(self.experiments)} experiments."
        )

    def __len__(self):
        return len(self.experiments)

    def __getitem__(self, idx):
        return self.experiments[idx], self.labels[idx]

    def get_experiment_id(self, idx):
        """Get the Experiment_ID for a given index."""
        return self.experiment_ids[idx]


class SegmentDataset3DSequenceWithMask(Dataset):
    """
    Dataset that groups timesteps by Experiment_ID into 3D tensors.
    Also returns a mask indicating which timesteps are valid (not padding).
    Labels are per-timestep with padding marked as -1.
    """

    def __init__(self, sensor_df, feature_cols):
        self.experiments = []
        self.labels = []
        self.masks = []
        self.experiment_ids = []

        self.unique_labels = sorted(sensor_df["Label"].unique())
        self.label_to_idx = {l: i for i, l in enumerate(self.unique_labels)}

        self.max_len = sensor_df.groupby("Experiment_ID").size().max()
        self.num_features = len(feature_cols)

        for exp_id, group in sensor_df.groupby("Experiment_ID"):
            values = group[feature_cols].values.astype(np.float32)
            seq_len = values.shape[0]

            padded_features = np.zeros(
                (self.max_len, self.num_features), dtype=np.float32
            )
            padded_features[:seq_len, :] = values

            padded_labels = np.full(self.max_len, -1, dtype=np.int64)
            label_indices = group["Label"].map(self.label_to_idx).values
            padded_labels[:seq_len] = label_indices

            mask = np.zeros(self.max_len, dtype=np.float32)
            mask[:seq_len] = 1.0

            self.experiments.append(torch.tensor(padded_features, dtype=torch.float32))
            self.labels.append(torch.tensor(padded_labels, dtype=torch.long))
            self.masks.append(torch.tensor(mask, dtype=torch.float32))
            self.experiment_ids.append(exp_id)
        logger.info(
            f"Created SegmentDataset3DSequenceWithMask with {len(self.experiments)} experiments."
        )

    def __len__(self):
        return len(self.experiments)

    def __getitem__(self, idx):
        return self.experiments[idx], self.labels[idx], self.masks[idx]

    def get_experiment_id(self, idx):
        """Get the Experiment_ID for a given index."""
        return self.experiment_ids[idx]
