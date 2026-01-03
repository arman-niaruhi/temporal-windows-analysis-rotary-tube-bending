import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks
from numpy.linalg import lstsq

def resample_experiment_fast(group, n=46, agg_metric="mean"):
    """
    Resample a single experiment to a fixed number of time bins using vectorized operations.
    
    Supports a wide range of aggregation metrics:
    - Basic statistics: mean, median, min, max, std, var, mad, rms
    - Shape metrics: skew, kurtosis
    - Signal metrics: energy, entropy, slope, fft_energy, fft_peakfreq
    - Distribution metrics: quantile, iqr, variation_ratio
    - Peaks: number of local maxima

    Args:
        group (pd.DataFrame): Sensor data for a single experiment.
        n (int): Number of time bins to resample into.
        agg_metric (str): Aggregation metric to compute per bin.

    Returns:
        pd.DataFrame: Resampled experiment data with one row per bin.
    """
    # Sort by time
    group = group.sort_values("Time_[s]")
    time_col = group["Time_[s]"].values

    # Create evenly spaced time bins
    time_bins = np.linspace(time_col.min(), time_col.max(), n + 1)
    bin_indices = np.digitize(time_col, time_bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n - 1)

    # Extract experiment ID
    exp_id = group["Experiment_ID"].iloc[0] if "Experiment_ID" in group.columns else group.name

    # Columns to aggregate (exclude time and ID)
    cols_to_process = [col for col in group.columns if col not in ["Time_[s]", "Experiment_ID"]]

    results = []

    for bin_idx in range(n):
        mask = bin_indices == bin_idx
        if not mask.any():  # Skip empty bins
            continue

        row_data = {"Experiment_ID": exp_id}
        t = time_col[mask]  # Time values in this bin

        for col in cols_to_process:
            values = group[col].values[mask]
            if len(values) == 0:
                continue

            # -------------------
            # Compute aggregation
            # -------------------
            if agg_metric == "mean":
                row_data[f"{col}_mean"] = values.mean()
            elif agg_metric == "median":
                row_data[f"{col}_median"] = np.median(values)
            elif agg_metric == "min":
                row_data[f"{col}_min"] = values.min()
            elif agg_metric == "max":
                row_data[f"{col}_max"] = values.max()
            elif agg_metric == "range":
                row_data[f"{col}_range"] = values.ptp()
            elif agg_metric == "std":
                row_data[f"{col}_std"] = values.std()
            elif agg_metric == "var":
                row_data[f"{col}_var"] = values.var()
            elif agg_metric == "mad":
                row_data[f"{col}_mad"] = np.abs(values - values.mean()).mean()
            elif agg_metric == "rms":
                row_data[f"{col}_rms"] = np.sqrt((values**2).mean())
            elif agg_metric == "skew":
                row_data[f"{col}_skew"] = 0.0 if values.std() < 1e-12 else skew(values)
            elif agg_metric == "kurtosis":
                row_data[f"{col}_kurtosis"] = 0.0 if values.std() < 1e-12 else kurtosis(values)
            elif agg_metric == "energy":
                row_data[f"{col}_energy"] = (values**2).sum()
            elif agg_metric == "entropy":
                abs_vals = np.abs(values)
                probs = abs_vals / (abs_vals.sum() + 1e-12)
                row_data[f"{col}_entropy"] = entropy(probs + 1e-12)
            elif agg_metric == "slope":
                # Linear regression slope per bin
                t_norm = t - t.mean()
                A = np.vstack([t_norm, np.ones_like(t_norm)]).T
                slope, intercept = lstsq(A, values, rcond=None)[0]
                row_data[f"{col}_slope"] = slope
                row_data[f"{col}_intercept"] = intercept
            elif agg_metric == "fft_energy":
                fft_vals = np.abs(np.fft.rfft(values))
                row_data[f"{col}_fft_energy"] = np.sum(fft_vals**2)
                row_data[f"{col}_fft_peakfreq"] = np.argmax(fft_vals)
            elif agg_metric == "quantile":
                q25, q75 = np.quantile(values, [0.25, 0.75])
                row_data[f"{col}_q25"] = q25
                row_data[f"{col}_q75"] = q75
                row_data[f"{col}_iqr"] = q75 - q25
            elif agg_metric == "peaks":
                peaks, _ = find_peaks(values)
                row_data[f"{col}_n_peaks"] = len(peaks)
            elif agg_metric == "variation_ratio":
                mu = values.mean()
                sigma = values.std()
                row_data[f"{col}_variation_ratio"] = sigma / (mu + 1e-12)

        results.append(row_data)

    return pd.DataFrame(results)


def resample_experiment_ultrafast(group, n=46, metric="mean"):
    """
    High-performance resampling using pandas groupby for standard metrics.
    
    For simple aggregation metrics (mean, median, std, var, etc.), this
    function is faster than fully vectorized resampling.

    Falls back to resample_experiment_fast for complex metrics.

    Args:
        group (pd.DataFrame): Sensor data for a single experiment.
        n (int): Number of bins.
        metric (str): Aggregation metric.

    Returns:
        pd.DataFrame: Resampled experiment with one row per bin.
    """
    group = group.sort_values("Time_[s]")
    time_col = group["Time_[s]"].values

    # Create bins and assign to _bin column
    time_bins = np.linspace(time_col.min(), time_col.max(), n + 1)
    group = group.copy()
    group["_bin"] = np.digitize(time_col, time_bins[:-1]) - 1
    group["_bin"] = group["_bin"].clip(0, n - 1)

    exp_id = group["Experiment_ID"].iloc[0] if "Experiment_ID" in group.columns else group.name

    # Columns eligible for aggregation
    cols_to_agg = [col for col in group.columns if col not in ["Time_[s]", "Experiment_ID", "_bin"]]

    agg_func_map = {
        "mean": "mean",
        "median": "median",
        "min": "min",
        "max": "max",
        "std": "std",
        "var": "var",
        "sum": "sum",
    }

    # Use fast pandas aggregation for simple metrics
    if metric in agg_func_map:
        result = group.groupby("_bin")[cols_to_agg].agg(agg_func_map[metric])
        result = result.add_suffix(f"_{metric}")
        result["Experiment_ID"] = exp_id
        return result.reset_index(drop=True)

    # Otherwise, fall back to extended fast resampling for complex metrics
    return resample_experiment_fast(group.drop("_bin", axis=1), n, agg_metric=metric)
