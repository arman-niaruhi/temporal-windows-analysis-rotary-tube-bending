import numpy as np
from ..base_utils.base_preprocessor import BasePreprocessor

class WindowAlgPreprocessor(BasePreprocessor):
    def normalize_angle(self, col="Angle[degree]ORDistance[mm]"):
        """Normalize a specific target column for window algorithm."""
        return self.normalize_column(col)

    def prepare_rf_data(self, X, Y):
        """
        Prepares data for RandomForestRegressor:
        Flattens sequences and appends normalized angle as extra feature.
        Returns X_rf, Y_rf as 2D arrays.
        """
        X_rf = []
        Y_rf = []

        num_samples, seq_len, num_features = X.shape
        num_angles = Y.shape[1]

        for sample_idx in range(num_samples):
            for angle_idx in range(num_angles):
                x_seq = X[sample_idx].flatten()
                degree = angle_idx / (num_angles - 1)  # normalized angle
                x_with_angle = np.append(x_seq, degree)
                X_rf.append(x_with_angle)
                Y_rf.append(Y[sample_idx, angle_idx])

        X_rf = np.array(X_rf)
        Y_rf = np.array(Y_rf)
        return X_rf, Y_rf