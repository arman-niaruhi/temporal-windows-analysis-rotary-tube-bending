import numpy as np

def validate_args(X, Y, sample_idx, angle_idx):
    if sample_idx < 0 or sample_idx >= X.shape[0]:
        raise IndexError(f"sample_idx {sample_idx} out of range")
    if angle_idx < 0 or angle_idx >= Y.shape[1]:
        raise IndexError(f"angle_idx {angle_idx} out of range")

def window_indices(seq_len, patch_size, stride, num_features):
    """Return list of index arrays for each feature window."""
    num_windows = (seq_len - patch_size) // stride + 1
    idx_list = []
    for w in range(num_windows):
        for f in range(num_features):
            start = w * stride * num_features + f
            idxs = np.arange(start, start + patch_size * num_features, num_features)
            idx_list.append((w, f, idxs))
    return idx_list
