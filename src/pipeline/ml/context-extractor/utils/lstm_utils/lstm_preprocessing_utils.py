import numpy as np
from ..base_utils.base_preprocessor import BasePreprocessor

class WindowAlgPreprocessor(BasePreprocessor):
    def normalize_angle(self, col="Angle[degree]ORDistance[mm]"):
        """Normalize a specific target column for window algorithm."""
        return self.normalize_column(col)
    
    def patchifier(self, X, window_size = 10):
        # Set the window size and also the number of windows
        num_samples, seq_len, num_features = X.shape
        num_windows = seq_len // window_size
        
        # Truncate extra timesteps if not divisible
        X_trunc = X[:,:num_windows * window_size,:]
        
        # Reshape to (samples, windows, window_size, features)
        X_reshaped = X_trunc.reshape(num_samples, num_windows, window_size, num_features)

        # Take mean over the window dimension
        X = X_reshaped.mean(axis=2)  # shape: (samples, num_windows, num_features)
        
        return X
