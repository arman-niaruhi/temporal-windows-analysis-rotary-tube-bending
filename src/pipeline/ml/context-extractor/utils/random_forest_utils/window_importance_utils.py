import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import shap
from ipywidgets import interact, IntSlider, Dropdown, SelectionSlider
import matplotlib.animation as animation
from utils.random_forest_utils.model import RandomForestTrainer
from utils.random_forest_utils.rf_preprocessing_utils import WindowAlgPreprocessor
class WindowImportance:
    def __init__(self, X, Y, rf, num_features, sensors_path, target_path):
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

        self.rf_preprocessor = WindowAlgPreprocessor(sensors_path=sensors_path, target_path=target_path)
        self.sensors_df, self.target_df = self.rf_preprocessor.read_data()
        self.sensors_df = self.rf_preprocessor.feature_selection()
        self.rf_preprocessor.normalize_angle()
        self.X = self.rf_preprocessor.group_and_pad(self.rf_preprocessor.sensor_df, group_col="Experiment_ID")
        self.Y = self.rf_preprocessor.group_and_pad(self.rf_preprocessor.target_df, group_col="Experiment_ID")[:,:,1:-1]

    # ---------------- FIT-style importance (target-specific) ----------------
    def window_fit_importance(self, sample_idx, angle_idx, patch_size, stride, mask_value=0.0):
        seq_len = self.X.shape[1]
        num_windows = (seq_len - patch_size) // stride + 1

        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree)

        y_base = self.rf.predict(x_with_angle.reshape(1, -1))[0][angle_idx]

        importance_scores = []
        for w in range(num_windows):
            start_w = w * stride * self.num_features
            end_w = start_w + patch_size * self.num_features

            if np.all(x_with_angle[start_w:end_w] == 0):
                importance_scores.append(np.nan)
                continue

            x_masked = x_with_angle.copy()
            x_masked[start_w:end_w] = mask_value

            y_hat = self.rf.predict(x_masked.reshape(1, -1))[0][angle_idx]
            importance_scores.append(np.abs(y_base - y_hat))

        return np.array(importance_scores)

    # ---------------- Permutation importance (target-specific) ----------------
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
            y_hat = self.rf.predict(x_perm.reshape(1, -1))[0][angle_idx]
            errors.append(np.linalg.norm(y_true - y_hat))
        return np.array(errors)

    # ---------------- Occlusion importance (target-specific) ----------------
    def window_occlusion_importance(self, sample_idx, angle_idx, patch_size, stride):
        seq_len = self.X.shape[1]
        num_windows = (seq_len - patch_size) // stride + 1
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree).reshape(1, -1)

        y_pred_full = self.rf.predict(x_with_angle)[0][angle_idx]

        errors = []
        for w in range(num_windows):
            start_w = w * stride * self.num_features
            end_w = start_w + patch_size * self.num_features

            if np.all(x_with_angle[0, start_w:end_w] == 0):
                errors.append(np.nan)
                continue

            x_masked = x_with_angle.copy()
            x_masked[0, start_w:end_w] = 0
            y_pred_masked = self.rf.predict(x_masked)[0][angle_idx]
            errors.append(np.abs(y_pred_full - y_pred_masked))
        return np.array(errors)

    # ---------------- WinIT importance (target-specific) ----------------
    def window_importance_winit(self, sample_idx, angle_idx, patch_size, stride, window_size=None, mask_value=0.0):
        seq_len = self.X.shape[1]
        num_features = self.num_features
        num_patches = (seq_len - patch_size) // stride + 1
        if window_size is None:
            window_size = patch_size

        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree)

        importance_scores = []

        for w in range(num_patches):
            start_w = w * stride * num_features
            end_w = start_w + patch_size * num_features

            if np.all(x_with_angle[start_w:end_w] == 0):
                importance_scores.append(np.nan)
                continue

            y_base = self.rf.predict(x_with_angle.reshape(1, -1))[0][angle_idx]

            x_masked = x_with_angle.copy()
            x_masked[start_w:end_w] = mask_value

            y_hat = self.rf.predict(x_masked.reshape(1, -1))[0][angle_idx]

            importance_scores.append(np.abs(y_base - y_hat))

        return np.array(importance_scores)

    # ---------------- SHAP importance (target-specific) ----------------
    def window_shap_importance(self, sample_idx, angle_idx, patch_size, stride):
        seq_len = self.X.shape[1]
        num_windows = (seq_len - patch_size) // stride + 1
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree).reshape(1, -1)

        explainer = shap.TreeExplainer(self.rf)
        shap_values = explainer.shap_values(x_with_angle)[0][:-1]  # drop angle

        importance_scores = []
        for w in range(num_windows):
            start_w = w * stride * self.num_features
            end_w = start_w + patch_size * self.num_features
            if np.all(x_with_angle[0, start_w:end_w] == 0):
                importance_scores.append(np.nan)
                continue
            importance_scores.append(np.sum(np.abs(shap_values[start_w:end_w])))
        return np.array(importance_scores)

    # ---------------- Noise importance (target-specific) ----------------
    def window_noise_importance(self, sample_idx, angle_idx, patch_size, stride, noise_std=0.5, n_repeats=5):
        seq_len = self.X.shape[1]
        num_windows = (seq_len - patch_size) // stride + 1
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree).reshape(1, -1)

        y_pred_full = self.rf.predict(x_with_angle)[0][angle_idx]

        importance_scores = []
        for w in range(num_windows):
            start_w = w * stride * self.num_features
            end_w = start_w + patch_size * self.num_features
            if np.all(x_with_angle[0, start_w:end_w] == 0):
                importance_scores.append(np.nan)
                continue

            diffs = []
            for _ in range(n_repeats):
                x_noisy = x_with_angle.copy()
                x_noisy[0, start_w:end_w] += np.random.normal(0, noise_std, end_w-start_w)
                y_pred_noisy = self.rf.predict(x_noisy)[0][angle_idx]
                diffs.append(np.abs(y_pred_full - y_pred_noisy))
            importance_scores.append(np.mean(diffs))
        return np.array(importance_scores)

    
    # ---------------- Window Importance Methods (WinIT style, corrected) ----------------
    def window_importance_winit(self, sample_idx, angle_idx, patch_size, stride, window_size=None, mask_value=0.0):
        """
        Compute window importance using WinIT (Windowed Importance in Time).

        Args:
            sample_idx: index of the sample in X
            angle_idx: index of the output angle / target
            patch_size: size of the window in timesteps
            stride: stride between windows
            window_size: N, number of future steps to consider (defaults to patch_size)
            mask_value: value used to mask the window (default 0.0)
        Returns:
            importance_scores: array of importance scores per window
        """
        seq_len = self.X.shape[1]
        num_features = self.num_features
        num_patches = (seq_len - patch_size) // stride + 1
        if window_size is None:
            window_size = patch_size

        # Flatten input and append degree
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree)

        importance_scores = []

        # Loop over sliding windows
        for w in range(num_patches):
            start_w = w * stride * num_features
            end_w = start_w + patch_size * num_features

            if np.all(x_with_angle[start_w:end_w] == 0):
                importance_scores.append(np.nan)
                continue

            # --- Step 1: compute baseline predictions for future window ---
            # Note: here we assume self.Y contains predictions for each future time step
            # You may need to loop over b in [w, w+window_size] depending on your model
            y_base_window = []
            for b in range(start_w, min(end_w + window_size * num_features, len(x_with_angle))):
                y_base = self.rf.predict(x_with_angle.reshape(1, -1))[0]
                y_base_window.append(y_base)
            y_base_window = np.array(y_base_window)

            # --- Step 2: mask the current window ---
            x_masked = x_with_angle.copy()
            x_masked[start_w:end_w] = mask_value

            # --- Step 3: compute predictions for masked window ---
            y_masked_window = []
            for b in range(start_w, min(end_w + window_size * num_features, len(x_with_angle))):
                y_hat = self.rf.predict(x_masked.reshape(1, -1))[0]
                y_masked_window.append(y_hat)
            y_masked_window = np.array(y_masked_window)

            # --- Step 4: compute temporal difference (WinIT) ---
            # Difference between [a,b] and [a+1,b] approximated as baseline vs masked
            diff = np.linalg.norm(y_base_window - y_masked_window, axis=-1)
            importance_scores.append(np.mean(diff))  # aggregate over future window

        return np.array(importance_scores)


    
    def window_permutation_error_heatmap(self, sample_idx, angle_idx, patch_size, stride):
        """
        Returns a (num_features, num_patches) array of permutation errors
        for each feature × window.
        """
        seq_len = self.X.shape[1]
        num_patches = (seq_len - patch_size) // stride + 1
        num_features = self.num_features

        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree)
        y_true = self.Y[sample_idx, angle_idx]

        errors = np.full((num_features, num_patches), np.nan)

        for w in range(num_patches):
            for f in range(num_features):
                # indices of this feature inside this window
                start_w = w * stride * num_features + f
                idxs = np.arange(start_w, start_w + patch_size * num_features, num_features)

                if np.all(x_with_angle[idxs] == 0):
                    continue

                x_perm = x_with_angle.copy()
                x_perm[idxs] = np.random.permutation(x_perm[idxs])
                y_hat = self.rf.predict(x_perm.reshape(1, -1))[0]
                errors[f, w] = np.linalg.norm(y_true - y_hat)

        return errors


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
    def plot_window_curve(X_window, df_sensors, df_target, importance_curve, experiment_id, angle_idx, patch_size, stride, title="Window Importance"):
        if experiment_id in list(df_sensors.Experiment_ID.unique()):
            unique_ids = np.unique(df_sensors["Experiment_ID"])
            sample_idx = np.where(unique_ids == experiment_id)[0][0]
        else:
            raise ValueError(
                f"Experiment ID '{experiment_id}' not found in experiment_ids."
            )

        # select the columns to plot
        cols = [col for col in df_sensors.columns if col != "Experiment_ID"]

        # select the rows corresponding to the sample
        df_sample = df_sensors[df_sensors["Experiment_ID"] == experiment_id][cols]

        x_axis = df_sample.index

        # Make sure it matches X_window
        if len(df_sample) != X_window.shape[0]:
            df_sample = df_sample.iloc[:X_window.shape[0], :]
        seq_len = df_sample.shape[0]
        num_features = df_sample.shape[1]
        num_windows = len(importance_curve)

        # Compute window start and end positions
        window_starts = np.array([w * stride for w in range(num_windows)])
        window_ends = window_starts + patch_size

        # Create combined figure
        fig, ax1 = plt.subplots(figsize=(18,6))

        # Plot all features
        for f in range(num_features):
            ax1.plot(np.arange(len(df_sample)), df_sample.iloc[:, f], label=cols[f])

        # Plot importance as rectangles
        ax2 = ax1.twinx()
        for start, end, imp in zip(window_starts, window_ends, importance_curve):
            ax2.fill_between([start, end], 0, imp, color='black', alpha=0.3)  # rectangle height = importance

        # Labels and titles
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Feature Value")
        ax2.set_ylabel("Importance (rescaled)")
        ax1.set_title(f"Sample {sample_idx}, Angle {angle_idx} - {title}")

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels() 
        lines2, labels2 = ax2.get_legend_handles_labels() 
        ax1.legend(lines + lines2, labels + labels2, loc='center left', bbox_to_anchor=(1.05, 0.5))

        plt.tight_layout()
        plt.show()
        
    def window_occlusion_importance_heatmap(self, sample_idx, angle_idx, patch_size, stride):
        seq_len = self.X.shape[1]
        num_patches = (seq_len - patch_size) // stride + 1
        num_features = self.num_features
        
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree).reshape(1, -1)
        y_pred_full = self.rf.predict(x_with_angle)[0]

        errors = np.full((num_features, num_patches), np.nan)

        for w in range(num_patches):
            for f in range(num_features):
                start_w = w * stride * num_features + f
                idxs = np.arange(start_w, start_w + patch_size * num_features, num_features)

                if np.all(x_with_angle[0, idxs] == 0):
                    continue

                x_masked = x_with_angle.copy()
                x_masked[0, idxs] = 0
                y_pred_masked = self.rf.predict(x_masked)[0]
                errors[f, w] = np.linalg.norm(y_pred_full - y_pred_masked)

        return errors


    def window_shap_importance_heatmap(self, sample_idx, angle_idx, patch_size, stride):
        seq_len = self.X.shape[1]
        num_patches = (seq_len - patch_size) // stride + 1
        num_features = self.num_features
        
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree).reshape(1, -1)

        explainer = shap.TreeExplainer(self.rf)
        shap_values = explainer.shap_values(x_with_angle)[0][:-1]  # drop angle feature

        errors = np.full((num_features, num_patches), np.nan)

        for w in range(num_patches):
            for f in range(num_features):
                start_w = w * stride * num_features + f
                idxs = np.arange(start_w, start_w + patch_size * num_features, num_features)

                if np.all(x_with_angle[0, idxs] == 0):
                    continue

                # absolute SHAP contribution for this feature × window
                errors[f, w] = np.sum(np.abs(shap_values[idxs]))

        return errors


    def window_noise_importance_heatmap(self, sample_idx, angle_idx, patch_size, stride, noise_std=0.5, n_repeats=5):
        seq_len = self.X.shape[1]
        num_patches = (seq_len - patch_size) // stride + 1
        num_features = self.num_features
        
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree).reshape(1, -1)
        y_pred_full = self.rf.predict(x_with_angle)[0]

        errors = np.full((num_features, num_patches), np.nan)

        for w in range(num_patches):
            for f in range(num_features):
                start_w = w * stride * num_features + f
                idxs = np.arange(start_w, start_w + patch_size * num_features, num_features)

                if np.all(x_with_angle[0, idxs] == 0):
                    continue

                diffs = []
                for _ in range(n_repeats):
                    x_noisy = x_with_angle.copy()
                    x_noisy[0, idxs] += np.random.normal(0, noise_std, len(idxs))
                    y_pred_noisy = self.rf.predict(x_noisy)[0]
                    diffs.append(np.linalg.norm(y_pred_full - y_pred_noisy))
                errors[f, w] = np.mean(diffs)

        return errors

        
        
    @staticmethod
    def plot_feature_heatmap(errors, df_sensors, experiment_id, angle_idx, patch_size, stride, title="Feature-wise Importance Heatmap"):
        """
        errors: (num_features, num_patches) array
        """
        num_features, num_patches = errors.shape
        window_centers = np.array([w * stride + patch_size//2 for w in range(num_patches)])

        # Extract feature names (skip Experiment_ID column)
        feature_names = [col for col in df_sensors.columns if col != "Experiment_ID"]

        plt.figure(figsize=(14, 6))
        im = plt.imshow(errors, aspect="auto", cmap="viridis",
                        extent=[window_centers[0], window_centers[-1], num_features-0.5, -0.5])

        plt.colorbar(im, label="Permutation Error")
        plt.xlabel("Timestep (window center)")
        plt.ylabel("Features")
        plt.title(f"Experiment {experiment_id}, Angle {angle_idx} - {title}")

        # Set y-ticks to feature names
        plt.yticks(ticks=np.arange(num_features), labels=feature_names)

        plt.show()

    


    # ---------------- Interactive Widget ----------------
    def interactive_window_plot(self, df_sensors, df_targets):
        importance_methods = {
            "Permutation": self.window_permutation_error,
            "Occlusion": self.window_occlusion_importance,
            "SHAP": self.window_shap_importance,
            "Noise": self.window_noise_importance,
            "FIT": self.window_fit_importance,  # <-- added FIT
            "Permutation (Heatmap)": self.window_permutation_error_heatmap,
            "Occlusion (Heatmap)": self.window_occlusion_importance_heatmap,
            "SHAP (Heatmap)": self.window_shap_importance_heatmap,
            "Noise (Heatmap)": self.window_noise_importance_heatmap
        }


        def plot_func(experiment_id, angle_idx, patch_size, stride, method):
            if experiment_id in list(df_sensors.Experiment_ID.unique()):
                unique_ids = np.unique(df_sensors["Experiment_ID"])
                sample_idx = np.where(unique_ids == experiment_id)[0][0]
            else:
                raise ValueError(
                    f"Experiment ID '{experiment_id}' not found in experiment_ids."
                )

            if "Heatmap" in method:
                errors = importance_methods[method](sample_idx, angle_idx, patch_size, stride)
                self.plot_feature_heatmap(errors, df_sensors, experiment_id, angle_idx, patch_size, stride, title=method)
            else:
                importance_curve = importance_methods[method](sample_idx, angle_idx, patch_size, stride)
                self.plot_window_curve(self.X[sample_idx], df_sensors, df_targets, importance_curve, experiment_id, angle_idx, patch_size, stride, title=method)


        sample_indices = [int(x) for x in list(df_sensors.Experiment_ID.unique())]
        interact(
            plot_func,
            experiment_id=SelectionSlider(
                options=sample_indices,
                value=sample_indices[0],
                description="Experiment ID"
            ),
            angle_idx=IntSlider(min=0, max=self.Y.shape[1]-2, step=1, value=0),
            patch_size=IntSlider(min=50, max=500, step=50, value=100),
            stride=IntSlider(min=10, max=200, step=10, value=10),
            method=Dropdown(options=list(importance_methods.keys()), value="Permutation", description="Method")
        )
        
    def make_importance_heatmap_video_over_angles(self, angle_indices, method, patch_size, stride,
                                              df_sensors, experiment_id, feature_idx_list, filename="importance_heatmap_over_angles.mp4"):
        """
        Create a video showing how feature × window heatmaps change over angles
        for one experiment, one setup, and one importance method, including sensor data subplot.
        """
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # Get all unique experiment IDs
        experiment_ids = df_sensors["Experiment_ID"].unique()

        # Suppose your target experiment ID is:
        target_id = 42  # replace with your experiment ID

        # Get the index in the array of unique IDs
        sample_idx = np.where(experiment_ids == target_id)[0][0]
        
        method_map = {
            "perm": self.window_permutation_error_heatmap,
            "occlusion": self.window_occlusion_importance_heatmap,
            "noise": self.window_noise_importance_heatmap,
            "shap": self.window_shap_importance_heatmap,
        }

        if method not in method_map:
            raise ValueError(f"Method '{method}' not supported. Choose from {list(method_map.keys())}.")
        
        importance_func = method_map[method]

        # Compute heatmaps
        heatmaps = []
        for a in angle_indices:
            X_rf, Y_rf = self.rf_preprocessor.prepare_rf_data(self.X, self.Y[:,a:a+2,:])
            trainer = RandomForestTrainer()
            trainer.train(X_rf, Y_rf)
            self.rf = trainer.model
            errors = importance_func(sample_idx, a, patch_size, stride)
            heatmaps.append(errors)

        # Feature names
        feature_names = [c for i,c in enumerate(df_sensors.columns) if c != "Experiment_ID" and i in feature_idx_list]

        # Axis extents
        num_features, num_patches = heatmaps[0].shape
        window_centers = np.array([w * stride + patch_size // 2 for w in range(num_patches)])
        extent = [window_centers[0], window_centers[-1], num_features - 0.5, -0.5]

        # Create figure with 2 subplots
        fig, (ax_heatmap, ax_sensors) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Initial heatmap
        im = ax_heatmap.imshow(heatmaps[0], aspect="auto", cmap="viridis", extent=extent)
        cbar = fig.colorbar(im, ax=ax_heatmap, label="Importance")
        ax_heatmap.set_xlabel("Timestep (window center)")
        ax_heatmap.set_ylabel("Features")
        ax_heatmap.set_yticks(np.arange(num_features))
        ax_heatmap.set_yticklabels(feature_names)
        ax_heatmap.set_title(f"Angle {angle_indices[0]} | Patch={patch_size}, Stride={stride}")

        # Sensor subplot (select relevant sensors by feature_idx_list)
        sensor_cols = [df_sensors.columns[i] for i in feature_idx_list if df_sensors.columns[i] != "Experiment_ID"]
        df_sample = df_sensors[df_sensors["Experiment_ID"] == experiment_id][sensor_cols].reset_index(drop=True)
        df_sample.plot(ax=ax_sensors, legend=True)
        ax_sensors.set_xlabel("Time")
        ax_sensors.set_ylabel("Sensor values")
        ax_sensors.set_title("Sensor data")

        # Update function for animation
        def update(frame):
            im.set_data(heatmaps[frame])
            ax_heatmap.set_title(f"Angle {angle_indices[frame]} | Patch={patch_size}, Stride={stride}")
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(heatmaps),
                                    interval=1000, blit=True, repeat=False)
        ani.save(filename, writer="ffmpeg", fps=1)
        plt.close(fig)
        print(f"🎥 Heatmap + sensors video saved as: {filename}")
