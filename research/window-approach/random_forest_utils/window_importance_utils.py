import numpy as np
from ipywidgets import interact, IntSlider, Dropdown, SelectionSlider, Checkbox
from typing import List, Tuple
from sklearn.ensemble import RandomForestRegressor

from utils.random_forest_utils.rf_preprocessing_utils import WindowAlgPreprocessor
from utils.random_forest_utils.window_importance_calculator import WindowImportanceCalculator
from utils.random_forest_utils.window_visualizer import plot_window_curve, plot_feature_heatmap


class WindowImportance:
    """
    WindowImportance
    ----------
    Orchestrates computation and visualization of window-based feature importance
    for sensor sequences using Occlusion, SHAP, and Noise methods.

    Attributes
    ----------
    rf : RandomForestRegressor
        Trained Random Forest model.
    num_features : int
        Number of features per timestep.
    rf_preprocessor : WindowAlgPreprocessor
        Handles reading, cleaning, and grouping sensor and target data.
    sensors_df : pd.DataFrame
        Preprocessed sensor data.
    target_df : pd.DataFrame
        Preprocessed target data.
    X : np.ndarray
        Grouped and padded sensor data, shape (n_samples, seq_len, n_features).
    Y : np.ndarray
        Grouped and padded target data, shape (n_samples, seq_len, n_targets).
    calculator : WindowImportanceCalculator
        Computes window importance for different methods.
    """

    def __init__(
        self,
        rf: RandomForestRegressor,
        num_features: int,
        sensors_path: str,
        target_path: str,
    ) -> None:
        self.rf = rf
        self.num_features = num_features
        self.rf_preprocessor = WindowAlgPreprocessor(sensors_path, target_path)

        self.sensors_df, self.target_df = self.rf_preprocessor.read_data()
        self.sensors_df = self.rf_preprocessor.feature_selection()
        self.rf_preprocessor.normalize_angle()

        self.X = self.rf_preprocessor.group_and_pad(
            self.rf_preprocessor.sensor_df, group_col="Experiment_ID"
        )
        self.Y = self.rf_preprocessor.group_and_pad(
            self.rf_preprocessor.target_df, group_col="Experiment_ID"
        )[:, :, 1:-1]

        self.calculator = WindowImportanceCalculator(rf, num_features)

    def _prepare_input(
        self, sample_idx: int, angle_idx: int
    ) -> Tuple[np.ndarray, float]:
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree).reshape(1, -1)
        return x_with_angle, degree

    def _compute_windows(
        self, seq_len: int, patch_size: int, stride: int
    ) -> List[Tuple[int, int]]:
        num_windows = (seq_len - patch_size) // stride + 1
        return [
            (
                w * stride * self.num_features,
                w * stride * self.num_features + patch_size * self.num_features,
            )
            for w in range(num_windows)
        ]

    def analyze_experiment(
        self,
        experiment_id: int,
        angle_idx: int,
        patch_size: int,
        stride: int,
        method: str = "Occlusion",
        heatmap: bool = False
    ):
        unique_ids = np.unique(self.sensors_df["Experiment_ID"])
        if experiment_id not in unique_ids:
            raise ValueError(f"Experiment ID '{experiment_id}' not found.")

        sample_idx = np.where(unique_ids == experiment_id)[0][0]
        df_sample = self.sensors_df[
            self.sensors_df["Experiment_ID"] == experiment_id
        ].iloc[:, 1:]

        x_with_angle, _ = self._prepare_input(sample_idx, angle_idx)
        seq_len = self.X.shape[1]
        windows = self._compute_windows(seq_len, patch_size, stride)

        if method == "Occlusion":
            importance = self.calculator.compute_occlusion_importance(x_with_angle, windows)
        elif method == "SHAP":
            importance = self.calculator.compute_shap_importance(x_with_angle, windows)
        elif method == "Noise":
            importance = self.calculator.compute_noise_importance(x_with_angle, windows)
        else:
            raise ValueError(f"Unknown method: {method}")

        if heatmap:
            if method == "Occlusion":
                errors = self.calculator.compute_occlusion_importance_matrix(x_with_angle, windows)
            elif method == "SHAP":
                errors = self.calculator.compute_shap_importance_matrix(x_with_angle, windows)
            elif method == "Noise":
                errors = self.calculator.compute_noise_importance_matrix(x_with_angle, windows)
            errors = np.nan_to_num(errors, nan=0.0)
            plot_feature_heatmap(
                errors=errors,
                feature_names=list(df_sample.columns),
                patch_size=patch_size,
                stride=stride,
                title=f"{method} Heatmap",
            )

        else:
            plot_window_curve(
                df_sample=df_sample,
                importance_curve=importance,
                patch_size=patch_size,
                stride=stride,
                title=f"{method} Curve",
            )

    def interactive_window_plot(self):
        importance_methods = ["Occlusion", "SHAP", "Noise"]

        def plot_func(experiment_id, angle_idx, patch_size, stride, method, heatmap):
            self.analyze_experiment(experiment_id, angle_idx, patch_size, stride, method, heatmap)

        sample_indices = [int(x) for x in list(self.sensors_df.Experiment_ID.unique())]

        interact(
            plot_func,
            experiment_id=SelectionSlider(
                options=sample_indices,
                value=sample_indices[0],
                description="Experiment ID",
            ),
            angle_idx=IntSlider(min=0, max=self.Y.shape[1] - 2, step=1, value=0),
            patch_size=IntSlider(min=50, max=500, step=50, value=50),
            stride=IntSlider(min=50, max=200, step=50, value=50),
            method=Dropdown(options=importance_methods, value="Occlusion", description="Method"),
            heatmap=Checkbox(value=False, description="Show Heatmap")
        )
