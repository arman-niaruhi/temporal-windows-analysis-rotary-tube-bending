import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import shap
from ipywidgets import interact, IntSlider, Dropdown, SelectionSlider

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

    # ---------------- FIT Importance ----------------
    def window_fit_importance(self, sample_idx, angle_idx, patch_size, stride, mask_value=0.0):
        """
        Compute FIT-style importance for each window:
        - Mask each window and compute the difference in model output
        - Returns importance score per window
        """
        seq_len = self.X.shape[1]
        num_windows = (seq_len - patch_size) // stride + 1

        # Flatten input and append degree
        x_input = self.X[sample_idx].flatten()
        degree = angle_idx / (self.Y.shape[1] - 1)
        x_with_angle = np.append(x_input, degree)

        # Baseline prediction
        y_base = self.rf.predict(x_with_angle.reshape(1, -1))[0]

        importance_scores = []
        for w in range(num_windows):
            start_w = w * stride * self.num_features
            end_w = start_w + patch_size * self.num_features

            if np.all(x_with_angle[start_w:end_w] == 0):
                importance_scores.append(np.nan)
                continue

            # Mask the window
            x_masked = x_with_angle.copy()
            x_masked[start_w:end_w] = mask_value

            # Predict and compute difference from baseline
            y_hat = self.rf.predict(x_masked.reshape(1, -1))[0]
            importance_scores.append(np.linalg.norm(y_base - y_hat))

        return np.array(importance_scores)

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
                print(1111111111111111)
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

        # Compute window centers
        window_centers = np.array([w * stride + patch_size//2 for w in range(num_windows)])

        # Create combined figure
        fig, ax1 = plt.subplots(figsize=(18,6))

        # Plot all features
        for f in range(num_features):
            ax1.plot(np.arange(len(df_sample)), df_sample.iloc[:, f], label=cols[f])

        # Plot rescaled importance on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(window_centers, importance_curve, color='black', linestyle='--', linewidth=2, label='Importance (rescaled)')

        # Labels and titles
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Feature Value")
        ax2.set_ylabel("Importance (rescaled)")
        ax1.set_title(f"Sample {sample_idx}, Angle {angle_idx} - {title}")

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='center left', bbox_to_anchor=(1.05, 0.5))

        plt.tight_layout(rect=[0,0,0.85,1])
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
            patch_size=IntSlider(min=50, max=500, step=50, value=200),
            stride=IntSlider(min=10, max=200, step=10, value=50),
            method=Dropdown(options=list(importance_methods.keys()), value="Permutation", description="Method")
        )

