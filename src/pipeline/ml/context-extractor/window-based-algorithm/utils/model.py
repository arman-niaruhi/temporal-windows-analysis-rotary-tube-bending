import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

'''
OOB Score (~0.78)
Out-of-bag (OOB) is like an internal cross-validation for Random Forests.
0.78 means the model explains roughly 78% of the variance on unseen samples from the training data.
This is a good sign that your model is not severely overfitting, since it’s close to your validation R².
Validation R² (~0.71)
This is the actual score on a held-out set.
0.71 means the model explains ~71% of the variance for truly unseen samples.
There’s a slight drop from OOB, which is normal. A drop of ~0.05–0.1 is acceptable.
Interpretation
The fact that OOB > Validation R² by a moderate margin indicates the model is generalizing fairly well.
If OOB >> Validation R², that would indicate overfitting.
If OOB << Validation R², that would suggest the OOB estimate might be unstable (rare).
'''

class RandomForestTrainer:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        max_features='sqrt',
        n_jobs=-1,
        oob_score=True,
        random_state=42,
        verbose=0,
        model_path="rf.joblib"
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose
        self.model_path = model_path
        self.model = None

    def train(self, X, Y):
        """Train the RandomForestRegressor on the provided data."""
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            oob_score=self.oob_score,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.model.fit(X, Y)
        print(f"Training complete. Model OOB score: {self.model.oob_score_ if self.oob_score else 'N/A'}")
        return self.model

    def save_model(self, path=None):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model trained yet. Call train() first.")
        path = path or self.model_path
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path=None):
        """Load a saved model from disk."""
        path = path or self.model_path
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model
    
    def predict_by_experiment_angle(self, sample_idx, X, Y):
        y_true = Y[sample_idx]  # Shape: (num_angles, 3)
        y_pred = []
        num_angles = Y.shape[1]
        output_size = Y.shape[2]

        for angle_idx in range(num_angles):
            x_seq = X[sample_idx].flatten()
            degree = angle_idx / (num_angles - 1)
            x_with_angle = np.append(x_seq, degree)
            y_hat = self.model.predict(x_with_angle.reshape(1, -1))
            y_pred.append(y_hat.flatten())

        y_pred = np.array(y_pred)  # Shape: (num_angles, 3)

        # ---------- Print comparison ----------
        # print(f"Comparing predictions for sample {sample_idx}:")
        # for angle_idx in range(num_angles):
        #     print(f"Angle {angle_idx:2d}: True={y_true[angle_idx]}, Pred={y_pred[angle_idx]}")

        # ---------- Optional: plot comparison ----------
        plt.figure(figsize=(12, 5))
        for dim in range(output_size):
            plt.plot(range(num_angles), y_true[:, dim], label=f'True dim {dim}')
            plt.plot(range(num_angles), y_pred[:, dim], '--', label=f'Pred dim {dim}')
        plt.xlabel('Angle index')
        plt.ylabel('Y value')
        plt.title(f'Predictions vs True for sample {sample_idx}')
        plt.legend()
        plt.show()

    
