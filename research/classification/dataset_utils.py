import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset class
class SegmentDatasetAllTimestamps(Dataset):
    def __init__(self, sensor_df, feature_cols):
        self.segments = []
        self.labels = []

        # Collect unique labels
        self.unique_labels = sorted(sensor_df["Label"].unique())
        self.label_to_idx = {l: i for i, l in enumerate(self.unique_labels)}

        for _, row in sensor_df.iterrows():
            feature_vector = row[feature_cols].values.astype(np.float32)  # ensure float32
            self.segments.append(torch.tensor(feature_vector, dtype=torch.float32))
            self.labels.append(self.label_to_idx[row["Label"]])

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]