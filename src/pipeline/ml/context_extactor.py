import pandas as pd
from src.logging.log_utils import log_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score


import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your system


class ContextExtractor():
    def __init__(self, input_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
        self.input_df = input_df
        self.target_df = target_df
        
    @log_function
    def extract_important_window(self, target_column: str,
                                method_num=1,
                                num_top_windows=1,
                                model_choice='xgboost'):

        X = np.array(self.input_df.drop(columns=["Experiment_ID"]).copy())
        cols_to_select = [col for col in self.target_df.columns 
                  if any(keyword in col for keyword in 
                         ["Angle","Secondary", "Main-axis", "Out-of-roundness", "Collapse"])]
        subset_df = self.target_df[cols_to_select].copy()
        y_all = np.array(subset_df)
        y = y_all[0]
        def split_non_overlapping(X, window_size):
            n_samples = X.shape[0]
            n_windows = n_samples // window_size
            return np.array(np.split(X[:n_windows*window_size], n_windows))
        
        windows = split_non_overlapping(X, window_size=50)
        


        # ---------------------------
        # 3. Choose and fit model
        # ---------------------------
        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "xgboost": XGBRegressor(n_estimators=100, random_state=42),
            "lightgbm": LGBMRegressor(n_estimators=100, random_state=42),
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.1),
            "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5),
            "svr": SVR(kernel='rbf', C=1.0),
            "mlp": MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            "adaboost": AdaBoostRegressor(n_estimators=100, random_state=42)
        }

        if model_choice not in models:
            print(f"Model {model_choice} not found. Using random_forest instead.")
            model_choice = "random_forest"

        model = models[model_choice]
        model.fit(X_agg, y)

        # Evaluate model performance
        y_pred = model.predict(X_agg)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f"Using model: {model_choice}")
        print(f"Model MAE: {mae:.4f}")
        print(f"Model R²: {r2:.4f}")

        # ---------------------------
        # 4. Compute feature window importance
        # ---------------------------
        feature_window_importance = []

        if method_num == 1:
            for i in range(n_timesteps):
                X_perm = X_agg.copy()
                X_perm[i, :] = np.random.permutation(X_perm[i, :])
                y_hat = model.predict(X_perm)
                imp = np.mean(np.abs(y - y_hat))
                feature_window_importance.append(imp)

        elif method_num == 2:
            if hasattr(model, 'feature_importances_'):
                mdi_importance = model.feature_importances_
                feature_window_importance = [np.sum(mdi_importance) for _ in range(n_timesteps)]
            else:
                raise ValueError(f"Model {model_choice} does not have feature_importances_ attribute")

        # ---------------------------
        # 5. Calculate importance percentages and find top windows
        # ---------------------------
        total_importance = np.sum(feature_window_importance)
        importance_percentages = (np.array(feature_window_importance) / total_importance) * 100
        top_window_indices = np.argsort(feature_window_importance)[-num_top_windows:][::-1]

        # ---------------------------
        # 6. Plot features with highlighted windows
        # ---------------------------
        timesteps = np.arange(n_timesteps)
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        colors = plt.cm.get_cmap('tab20', n_features)(np.arange(n_features))
        for f in range(n_features):
            axes[0].plot(timesteps, X_agg[:, f], marker='.', markersize=2, color=colors[f],
                        linestyle='', alpha=0.6, label=f'Feature {f+1}')

        highlight_colors = ['red', 'orange', 'purple', 'green', 'brown', 'cyan', 'magenta', 'yellow']
        for i, window_idx in enumerate(top_window_indices):
            if i < len(highlight_colors):
                axes[0].axvspan(window_idx, window_idx + 1,
                                color=highlight_colors[i], alpha=0.4,
                                label=f'Top {i+1} window ({importance_percentages[window_idx]:.1f}%)')
                axes[0].text(window_idx + 0.5, axes[0].get_ylim()[1] * 0.95,
                            f'{importance_percentages[window_idx]:.1f}%',
                            ha='center', va='top', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=highlight_colors[i], alpha=0.8))

        axes[0].set_ylabel("Feature values")
        axes[0].set_title("Feature values with top important windows highlighted")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(timesteps, y, marker='.', markersize=3, color='blue', linestyle='-', alpha=0.8, linewidth=0.5, label='Target values')
        axes[1].set_xlabel("Target index (degree)")
        axes[1].set_ylabel("Target")
        axes[1].set_title("Target values")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # ---------------------------
        # 7. Optional: Plot importance bar and pie charts
        # ---------------------------
        plt.figure(figsize=(14, 8))

        # Bar chart importance
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(n_timesteps), feature_window_importance, color='skyblue', alpha=0.7)
        for i, window_idx in enumerate(top_window_indices):
            if i < len(highlight_colors):
                bars[window_idx].set_color(highlight_colors[i])
                bars[window_idx].set_alpha(0.8)
        plt.xlabel('Window Number')
        plt.ylabel('Importance Score')
        plt.title('Feature Window Importance Scores')
        plt.grid(axis='y', alpha=0.3)

        # Bar chart percentages
        plt.subplot(2, 2, 2)
        bars_pct = plt.bar(range(n_timesteps), importance_percentages, color='lightcoral', alpha=0.7)
        for i, window_idx in enumerate(top_window_indices):
            if i < len(highlight_colors):
                bars_pct[window_idx].set_color(highlight_colors[i])
                bars_pct[window_idx].set_alpha(0.8)
        plt.xlabel('Window Number')
        plt.ylabel('Importance Percentage (%)')
        plt.title('Importance Percentage by Window')
        plt.grid(axis='y', alpha=0.3)

        # Pie chart
        plt.subplot(2, 2, (3, 4))
        labels = [f'W{w}\n({importance_percentages[w]:.1f}%)' for w in range(n_timesteps)]
        colors_pie = ['lightgray'] * n_timesteps
        for i, window_idx in enumerate(top_window_indices):
            if i < len(highlight_colors):
                colors_pie[window_idx] = highlight_colors[i]
        wedges, texts, autotexts = plt.pie(importance_percentages, labels=labels, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90, labeldistance=1.1)
        plt.axis('equal')
        plt.title('Importance Distribution by Window (%)')
        for text in texts + autotexts:
            text.set_fontsize(8)

        plt.tight_layout()
        plt.show()
