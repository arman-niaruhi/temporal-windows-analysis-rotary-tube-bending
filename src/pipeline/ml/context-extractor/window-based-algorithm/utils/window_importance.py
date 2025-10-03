import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import shap
from ipywidgets import interact, IntSlider, Dropdown

class WindowImportance:
    def __init__(self, X, Y, rf, num_features):
        """
        X: (samples, seq_len, features)
        Y: (samples, angles, channels)
        rf: trained RandomForest model
        num_features: number of features per timestep
        """
        self.X = X
        self.Y = Y
        self.rf = rf
        self.num_features = num_features

    # ---------------- Window Importance Methods ----------------
    def window_permutation_error(self, sample_idx, angle_idx, patch_size, stride):
        seq_len = self.X.shape[1]
        num_patches = (seq_len - patch_size) // stride + 1
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree)
        y_true = self.Y[sample_idx, angle_idx]

        errors = []
        for w in range(num_patches):
            start_w = w * stride * self.num_features
            end_w = start_w + patch_size * self.num_features

            if np.all(x_with_angle[start_w:end_w] == 0):
                errors.append(np.nan)
                continue

            x_perm = x_with_angle.copy()
            x_perm[start_w:end_w] = np.random.permutation(x_perm[start_w:end_w])
            y_hat = self.rf.predict(x_perm.reshape(1, -1))[0]
            errors.append(np.linalg.norm(y_true - y_hat))
        return np.array(errors)

    def window_occlusion_importance(self, sample_idx, angle_idx, patch_size, stride):
        seq_len = self.X.shape[1]
        num_windows = (seq_len - patch_size) // stride + 1
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree).reshape(1, -1)
        y_pred_full = self.rf.predict(x_with_angle)[0]

        errors = []
        for w in range(num_windows):
            start_w = w * stride * self.num_features
            end_w = start_w + patch_size * self.num_features

            if np.all(x_with_angle[0, start_w:end_w] == 0):
                errors.append(np.nan)
                continue

            x_masked = x_with_angle.copy()
            x_masked[0, start_w:end_w] = 0
            y_pred_masked = self.rf.predict(x_masked)[0]
            errors.append(np.linalg.norm(y_pred_full - y_pred_masked))
        return np.array(errors)

    def window_shap_importance(self, sample_idx, angle_idx, patch_size, stride):
        seq_len = self.X.shape[1]
        num_windows = (seq_len - patch_size) // stride + 1
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree).reshape(1, -1)

        explainer = shap.TreeExplainer(self.rf)
        shap_values = explainer.shap_values(x_with_angle)[0][:-1]  # drop angle feature

        importance_scores = []
        for w in range(num_windows):
            start_w = w * stride * self.num_features
            end_w = start_w + patch_size * self.num_features
            if np.all(x_with_angle[0, start_w:end_w] == 0):
                importance_scores.append(np.nan)
                continue
            importance_scores.append(np.sum(np.abs(shap_values[start_w:end_w])))
        return np.array(importance_scores)

    def window_noise_importance(self, sample_idx, angle_idx, patch_size, stride, noise_std=0.5, n_repeats=5):
        seq_len = self.X.shape[1]
        num_windows = (seq_len - patch_size) // stride + 1
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree).reshape(1, -1)
        y_pred_full = self.rf.predict(x_with_angle)[0]

        importance_scores = []
        for w in range(num_windows):
            start_w = w * stride * self.num_features
            end_w   = start_w + patch_size * self.num_features
            if np.all(x_with_angle[0, start_w:end_w] == 0):
                importance_scores.append(np.nan)
                continue

            diffs = []
            for _ in range(n_repeats):
                x_noisy = x_with_angle.copy()
                x_noisy[0, start_w:end_w] += np.random.normal(0, noise_std, end_w-start_w)
                y_pred_noisy = self.rf.predict(x_noisy)[0]
                diffs.append(np.linalg.norm(y_pred_full - y_pred_noisy))
            importance_scores.append(np.mean(diffs))
        return np.array(importance_scores)

    # ---------------- Plot Function ----------------
    @staticmethod
    def plot_window_curve(X_window, df_sensors,df_target, importance_curve, sample_idx, angle_idx, patch_size, stride, title="Window Importance"):
        _, ax1 = plt.subplots(figsize=(12,6))

        # select the columns to plot
        cols = [col for col in df_sensors.columns if col != "Experiment_ID"]
        
        # select the rows corresponding to the sample
        df_sample = df_sensors[df_sensors["Experiment_ID"]==sample_idx][cols]
        
        # Make sure it matches X_window
        if len(df_sample) != X_window.shape[0]:
            # either slice or interpolate
            df_sample = df_sample.iloc[:X_window.shape[0], :]

        # plot each feature
        for i, col in enumerate(cols):
            ax1.plot(df_sample.index, df_sample[col].values, label=col)
        
        ax2 = ax1.twinx()
        num_windows = len(importance_curve)
        window_centers = np.array([w * stride + patch_size//2 for w in range(num_windows)])
        ax2.plot(window_centers, importance_curve, color='black', linestyle='--', linewidth=2, label='Importance')

        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Feature Value")
        ax2.set_ylabel("Importance Score")
        ax1.set_title(f"Sample {sample_idx}, Angle {angle_idx} - {title}")
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.show()



    # ---------------- Interactive Widget ----------------
    def interactive_window_plot(self, df_sensors, df_targets):
        importance_methods = {
            "Permutation": self.window_permutation_error,
            "Occlusion": self.window_occlusion_importance,
            "SHAP": self.window_shap_importance,
            "Noise": self.window_noise_importance
        }

        def plot_func(sample_idx, angle_idx, patch_size, stride, method):
            importance_curve = importance_methods[method](sample_idx, angle_idx, patch_size, stride)
            self.plot_window_curve(self.X[sample_idx], df_sensors, df_targets, importance_curve, sample_idx, angle_idx, patch_size, stride, title=method)

        interact(
            plot_func,
            sample_idx=IntSlider(min=0, max=self.X.shape[0]-1, step=1, value=0),
            angle_idx=IntSlider(min=0, max=self.Y.shape[1]-2, step=1, value=0),
            patch_size=IntSlider(min=50, max=500, step=50, value=200),
            stride=IntSlider(min=10, max=200, step=10, value=50),
            method=Dropdown(options=list(importance_methods.keys()), value="Permutation", description="Method")
        )
