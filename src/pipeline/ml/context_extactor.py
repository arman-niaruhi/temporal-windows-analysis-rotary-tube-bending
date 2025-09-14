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

class ContextExtractor():
    def __init__(self, input_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
        self.input_df = input_df
        self.target_df = target_df
        
    @log_function
    def extract_important_window(self, target_column:str,
                                 window_size = 200,
                                 method_num = 1,
                                 num_top_windows = 1,
                                 target_window_idx = 0,
                                 model_choice = 'xgboost'):
        X = np.array(self.input_df.drop(columns=["Experiment_ID"]).copy())
        y = np.array(self.target_df[target_column].copy())
        n_timesteps = len(y)
        n_features = X.shape[1]

        n_feature_windows = n_timesteps // window_size
        target_start = target_window_idx * window_size
        target_end = target_start + window_size
        target_window = y[target_start:target_end]

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
        model.fit(X, y)

        # Evaluate model performance
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"Using model: {model_choice}")
        print(f"Model MAE: {mae:.4f}")
        print(f"Model R²: {r2:.4f}")

        # ---------------------------
        # 4. Compute feature window importance
        # ---------------------------
        feature_window_importance = []

        match method_num:
            case 1:
                for w in range(n_feature_windows):
                    start = w * window_size
                    end = start + window_size

                    X_perm = X.copy()
                    for f in range(n_features):
                        X_perm[start:end, f] = np.random.permutation(X_perm[start:end, f])

                    y_hat = model.predict(X_perm)
                    imp = np.mean(np.abs(y - y_hat))
                    feature_window_importance.append(imp)
            case 2:
                mdi_importance = model.feature_importances_
                for w in range(n_feature_windows):
                    feature_window_importance.append(np.sum(mdi_importance))

        # ---------------------------
        # 5. Calculate importance percentages and find top windows
        # ---------------------------
        total_importance = np.sum(feature_window_importance)
        importance_percentages = (np.array(feature_window_importance) / total_importance) * 100

        top_window_indices = np.argsort(feature_window_importance)[-num_top_windows:][::-1]

        for i, window_idx in enumerate(top_window_indices):
            importance = feature_window_importance[window_idx]
            percentage = importance_percentages[window_idx]
            start_time = window_idx * window_size
            end_time = start_time + window_size

        # ---------------------------
        # 6. Plot with multiple important windows highlighted ON TOP of feature values
        # ---------------------------
        timesteps = np.arange(n_timesteps)
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        # Use a colormap that can handle any number of features
        colors = plt.cm.get_cmap('tab20', n_features)(np.arange(n_features))

        # Plot features with smaller points and no lines
        for f in range(n_features):
            axes[0].plot(timesteps, X[:, f], marker='.', markersize=1, color=colors[f], 
                        linestyle='', alpha=0.6, label=f'Feature {f+1}')

        # Highlight top important windows with different colors (ON TOP of the features)
        highlight_colors = ['red', 'orange', 'purple', 'green', 'brown', 'cyan', 'magenta', 'yellow']
        for i, window_idx in enumerate(top_window_indices):
            if i < len(highlight_colors):
                feature_start = window_idx * window_size
                feature_end = feature_start + window_size
                
                # Create semi-transparent rectangles for better visibility
                axes[0].axvspan(feature_start, feature_end, 
                            color=highlight_colors[i], alpha=0.4, 
                            label=f'Top {i+1} window ({importance_percentages[window_idx]:.1f}%)')
                
                # Add text annotation for each window
                mid_point = (feature_start + feature_end) / 2
                y_range = axes[0].get_ylim()
                text_y = y_range[1] - 0.05 * (y_range[1] - y_range[0]) * (i + 1)
                axes[0].text(mid_point, text_y, f'{importance_percentages[window_idx]:.1f}%',
                            ha='center', va='center', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=highlight_colors[i], alpha=0.8))

        axes[0].set_ylabel("Feature values")
        axes[0].set_title("Feature values with top important windows highlighted (small points, no lines)")
        #axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Plot target with smaller points
        axes[1].plot(timesteps, y, marker='.', markersize=2, color='blue', 
                    linestyle='-', alpha=0.8, linewidth=0.5, label='Target values')
        axes[1].axvspan(target_start, target_end, color='grey', alpha=0.4, label='Selected target window')
        axes[1].set_xlabel("Time steps")
        axes[1].set_ylabel("Target")
        axes[1].set_title("Target values with selected window highlighted")
        #axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Additional analysis: Show the importance values for all windows
        plt.figure(figsize=(14, 8))

        # Bar chart with importance values
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(n_feature_windows), feature_window_importance, color='skyblue', alpha=0.7)
        plt.xlabel('Window Number')
        plt.ylabel('Importance Score')
        plt.title('Feature Window Importance Scores')
        plt.xticks(range(n_feature_windows))
        plt.grid(axis='y', alpha=0.3)

        # Highlight top windows in bar chart
        for i, window_idx in enumerate(top_window_indices):
            if i < len(highlight_colors):
                bars[window_idx].set_color(highlight_colors[i])
                bars[window_idx].set_alpha(0.8)

        # Bar chart with percentages
        plt.subplot(2, 2, 2)
        bars_pct = plt.bar(range(n_feature_windows), importance_percentages, color='lightcoral', alpha=0.7)
        plt.xlabel('Window Number')
        plt.ylabel('Importance Percentage (%)')
        plt.title('Importance Percentage by Window')
        plt.xticks(range(n_feature_windows))
        plt.grid(axis='y', alpha=0.3)

        # Highlight top windows in percentage chart
        for i, window_idx in enumerate(top_window_indices):
            if i < len(highlight_colors):
                bars_pct[window_idx].set_color(highlight_colors[i])
                bars_pct[window_idx].set_alpha(0.8)

        # Pie chart showing percentage distribution
        plt.subplot(2, 2, (3, 4))
        # Create labels with both window number and percentage
        labels = [f'W{w}\n({importance_percentages[w]:.1f}%)' for w in range(n_feature_windows)]
        colors_pie = ['lightgray'] * n_feature_windows
        # Highlight top windows in pie chart
        for i, window_idx in enumerate(top_window_indices):
            if i < len(highlight_colors):
                colors_pie[window_idx] = highlight_colors[i]

        wedges, texts, autotexts = plt.pie(importance_percentages, labels=labels, colors=colors_pie, 
                                        autopct='%1.1f%%', startangle=90, labeldistance=1.1)
        plt.axis('equal')
        plt.title('Importance Distribution by Window (%)')

        # Make text smaller for better readability
        for text in texts + autotexts:
            text.set_fontsize(8)

        plt.tight_layout()
        plt.show()