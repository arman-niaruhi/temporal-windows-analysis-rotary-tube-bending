import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor


class WindowImportanceCalculator:
    """
    WindowImportanceCalculator
    --------------------------
    Calculates window-based importance scores for input sequences using
    multiple interpretability methods:
      - Occlusion: mask sections of the input and measure prediction change
      - SHAP: compute SHAP value magnitudes for each input segment
      - Noise: inject random noise and measure sensitivity

    Attributes
    ----------
    rf : RandomForestRegressor
        Trained random forest regressor used for prediction and SHAP explanations.

    num_features : int
        Number of input features per time step.
    """

    def __init__(self, rf: RandomForestRegressor, num_features: int):
        """
        Initialize the WindowImportanceCalculator.

        Parameters
        ----------
        rf : RandomForestRegressor
            Trained Random Forest model.

        num_features : int
            Number of input features per timestep.
        """
        self.rf = rf
        self.num_features = num_features

    # ---------------- Helper Functions ----------------
    def _predict_with_mask(
        self, x_input: np.ndarray, start: int, end: int
    ) -> np.ndarray:
        """
        Predict output after masking a segment of input to zero.
        """
        x_masked = x_input.copy()
        x_masked[0, start:end] = 0
        return self.rf.predict(x_masked)[0]

    def _predict_with_noise(
        self,
        x_input: np.ndarray,
        start: int,
        end: int,
        noise_std: float,
        n_repeats: int,
    ) -> float:
        """
        Predict multiple times after injecting Gaussian noise into a segment.
        """
        y_pred_full = self.rf.predict(x_input)[0]
        diffs = []
        for _ in range(n_repeats):
            x_noisy = x_input.copy()
            x_noisy[0, start:end] += np.random.normal(0, noise_std, end - start)
            y_pred_noisy = self.rf.predict(x_noisy)[0]
            diffs.append(np.linalg.norm(y_pred_full - y_pred_noisy))
        return np.mean(diffs)

    # ---------------- Importance Methods ----------------
    def compute_occlusion_importance(self, x_with_angle, windows) -> np.ndarray:
        """
        Compute occlusion-based window importance by masking windows.
        """
        y_pred_full = self.rf.predict(x_with_angle)[0]
        errors = []
        for start, end in windows:
            if np.all(x_with_angle[0, start:end] == 0):
                errors.append(np.nan)
                continue
            y_pred_masked = self._predict_with_mask(x_with_angle, start, end)
            errors.append(np.linalg.norm(y_pred_full - y_pred_masked))
        return np.array(errors)

    def compute_shap_importance(self, x_with_angle, windows) -> np.ndarray:
        """
        Compute SHAP-based window importance by summing absolute SHAP values
        within each window.
        """
        explainer = shap.TreeExplainer(self.rf)
        shap_values = explainer.shap_values(x_with_angle)[0][:-1]
        importance_scores = []
        for start, end in windows:
            if np.all(x_with_angle[0, start:end] == 0):
                importance_scores.append(np.nan)
                continue
            importance_scores.append(np.sum(np.abs(shap_values[start:end])))
        return np.array(importance_scores)

    def compute_noise_importance(
        self, x_with_angle, windows, noise_std: float = 0.5, n_repeats: int = 5
    ) -> np.ndarray:
        """
        Compute noise-based window importance by injecting Gaussian noise
        and measuring prediction change.
        """
        scores = []
        for start, end in windows:
            if np.all(x_with_angle[0, start:end] == 0):
                scores.append(np.nan)
                continue
            score = self._predict_with_noise(
                x_with_angle, start, end, noise_std, n_repeats
            )
            scores.append(score)
        return np.array(scores)
    
    def compute_occlusion_importance_matrix(self, x_with_angle, windows) -> np.ndarray:
        """
        Compute occlusion-based importance for all features as a matrix.
        Rows correspond to features, columns correspond to windows.

        Parameters
        ----------
        x_with_angle : np.ndarray
            Input sample of shape [1, seq_len].
        windows : list of tuples
            List of (start, end) indices defining the windows.

        Returns
        -------
        np.ndarray
            Importance matrix of shape (num_features, num_windows).
        """
        y_pred_full = self.rf.predict(x_with_angle)[0]
        num_features = self.num_features
        num_windows = len(windows)
        
        importance_matrix = np.full((num_features, num_windows), np.nan)

        for w, (start, end) in enumerate(windows):
            for f in range(num_features):
                # compute indices for this feature in the current window
                idxs = np.arange(start + f, end, num_features)
                if np.all(x_with_angle[0, idxs] == 0):
                    continue
                y_pred_masked = self._predict_with_mask(x_with_angle, idxs[0], idxs[-1] + 1)
                importance_matrix[f, w] = np.linalg.norm(y_pred_full - y_pred_masked)

        return importance_matrix

    def compute_shap_importance_matrix(self, x_with_angle, windows) -> np.ndarray:
        """
        Compute SHAP-based importance for all features as a matrix.
        Rows correspond to features, columns correspond to windows.
        """
        explainer = shap.TreeExplainer(self.rf)
        shap_values = explainer.shap_values(x_with_angle)[0]  # shape: (1, seq_len)
        num_features = self.num_features
        num_windows = len(windows)
        
        importance_matrix = np.full((num_features, num_windows), np.nan)

        for w, (start, end) in enumerate(windows):
            for f in range(num_features):
                idxs = np.arange(start + f, end, num_features)
                if np.all(x_with_angle[0, idxs] == 0):
                    continue
                importance_matrix[f, w] = np.sum(np.abs(shap_values[idxs]))

        return importance_matrix

    def compute_noise_importance_matrix(self, x_with_angle, windows, noise_std=0.5, n_repeats=5) -> np.ndarray:
        """
        Compute noise-based importance for all features as a matrix.
        Rows correspond to features, columns correspond to windows.
        """
        y_pred_full = self.rf.predict(x_with_angle)[0]
        num_features = self.num_features
        num_windows = len(windows)
        
        importance_matrix = np.full((num_features, num_windows), np.nan)

        for w, (start, end) in enumerate(windows):
            for f in range(num_features):
                idxs = np.arange(start + f, end, num_features)
                if np.all(x_with_angle[0, idxs] == 0):
                    continue
                diffs = []
                for _ in range(n_repeats):
                    x_noisy = x_with_angle.copy()
                    x_noisy[0, idxs] += np.random.normal(0, noise_std, len(idxs))
                    y_pred_noisy = self.rf.predict(x_noisy)[0]
                    diffs.append(np.linalg.norm(y_pred_full - y_pred_noisy))
                importance_matrix[f, w] = np.mean(diffs)

        return importance_matrix

