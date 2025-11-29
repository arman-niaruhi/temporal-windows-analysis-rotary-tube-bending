import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import rcParams


class OrganizedImageSaver:
    def __init__(self, base_dir="images", machine_part="COMPLETE"):
        self.base_dir = Path(base_dir)

        self.machine_part = machine_part

        # Create four main folders
        self.predictions_dir = self.base_dir / "01_predictions"
        self.loss_dir = self.base_dir / "02_loss"
        self.attention_dir = self.base_dir / "03_attention"
        self.attention_csv_dir = self.base_dir / "03_attention_csv"

        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.loss_dir.mkdir(parents=True, exist_ok=True)
        self.attention_dir.mkdir(parents=True, exist_ok=True)
        self.attention_csv_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_count = 0

    def save_epoch_plots(
        self,
        sensor_data,
        feature_names,
        output_feature_names,
        pred_data,
        loss_data,
        attn_data,
        epoch,
        x_axis,
        y_lim,
        PREDICTIONS_OUT,
        train_loss,
        val_loss,
        best_val_loss,
        annot_timesteps,
    ):
        """Save each subplot as a separate image in organized folders"""

        plt.style.use("tableau-colorblind10")

        true_np, pred_np, idxs = pred_data
        num_samples = len(idxs)
        n_features = true_np.shape[-1]

        # Horizontal layout: one row per sample, one column per feature
        nrows = num_samples
        ncols = n_features

        fig_pred, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5 * ncols, 3.5 * nrows),
            sharex=True,
            sharey=False,
        )

        axes = np.array(axes).reshape(nrows, ncols)

        for row_i, idx in enumerate(idxs):
            for feat in range(n_features):
                ax = axes[row_i, feat]

                ax.plot(
                    x_axis,
                    true_np[idx, :, feat],
                    "o-",
                    lw=2.2,
                    ms=4,
                    label="True Value",
                )

                ax.plot(
                    x_axis,
                    pred_np[idx, :, feat],
                    "--s",
                    lw=1.8,
                    ms=4,
                    alpha=0.9,
                    label="Prediction",
                )

                ax.set_ylim(*y_lim)
                ax.grid(True, linestyle=":", alpha=0.55)

                # Row labels (left-most column)
                if feat == 0:
                    ax.set_ylabel(f"Sample {row_i}", fontsize=12, weight="bold")

                # Column labels (top row)
                if row_i == 0:
                    ax.set_title(output_feature_names[feat], fontsize=13, weight="bold")

                # Legend only once per row
                if feat == n_features - 1:
                    ax.legend(fontsize=9, loc="upper right")

        # Common labels
        fig_pred.suptitle(
            f"Predictions – Epoch {epoch} ({num_samples} samples × {n_features} features)",
            fontsize=16,
            weight="bold",
        )

        fig_pred.supxlabel(f"Prediction Index (Total: {PREDICTIONS_OUT})", fontsize=13)
        fig_pred.supylabel("Target Value", fontsize=13)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        pred_path = self.predictions_dir / f"predictions_epoch_{epoch:04d}.png"
        fig_pred.savefig(pred_path, dpi=180, bbox_inches="tight")
        plt.close(fig_pred)

        # 2. LOSS PLOT
        fig_loss = plt.figure(figsize=(10, 7))
        ax_loss = fig_loss.add_subplot(111)

        epochs_list, val_losses, train_losses = loss_data

        ax_loss.plot(
            epochs_list,
            train_losses,
            color="#1f77b4",
            lw=3,
            alpha=0.7,
            label="Train MSE",
        )
        ax_loss.plot(epochs_list, val_losses, color="#d62728", lw=3, label="Val MSE")
        ax_loss.plot(
            epochs_list,
            [best_val_loss] * len(epochs_list),
            color="green",
            lw=2.5,
            ls="--",
            label="Best Val MSE",
        )

        ax_loss.set_xlabel("Epoch", fontsize=12)
        ax_loss.set_ylabel("MSE", fontsize=12)
        ax_loss.set_title(
            f"Training Progress - Epoch {epoch}\nTrain: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f}",
            fontweight="bold",
            fontsize=14,
        )
        ax_loss.grid(alpha=0.3)
        ax_loss.legend(fontsize=10)
        plt.tight_layout()

        loss_path = self.loss_dir / f"loss_epoch_{epoch:04d}.png"
        fig_loss.savefig(loss_path, dpi=150, bbox_inches="tight")
        plt.close(fig_loss)

        # 3. ATTENTION HEATMAP
        attn_mean = attn_data
        attn_path = self.attention_dir / f"attention_epoch_{epoch:04d}.png"
        self.plot_selected_features_with_attn_heatmap(
            sensor_data, feature_names, attn_mean, attn_path, annot_timesteps
        )
        attn_df = pd.DataFrame(
            attn_mean,
            index=[f"Pred_{i}" for i in range(attn_mean.shape[0])],
            columns=[f"Time_{i}" for i in range(attn_mean.shape[1])],
        )

        csv_path = self.attention_csv_dir / f"attention_epoch_{epoch:04d}.csv"
        attn_df.to_csv(csv_path, float_format="%.6f")

        self.epoch_count = epoch

        return pred_path, loss_path, attn_path, csv_path

    def plot_selected_features_with_attn_heatmap(
        self,
        sensor_data,
        sensor_names,
        attn_mean,
        attn_path,
        annot_timesteps=None,
        sample_idx=100,
        figsize=(25, 12),
    ):
        """
        Plots selected features with attention heatmap at the bottom.
        Includes legend for the top plot on the right side.
        Enhanced with beautiful styling and improved aesthetics.
        """
        # Set beautiful style parameters
        rcParams["font.family"] = "sans-serif"
        rcParams["font.size"] = 10

        # Remove '_mean' from all feature names
        cleaned_feature_names = [name.replace("_mean", "") for name in sensor_names]

        # Create figure with subplots
        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.clf()
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.25, wspace=0.3)
        ax_main = fig.add_subplot(gs[0])
        ax_heatmap = fig.add_subplot(gs[1])

        # Get the data for the selected sample
        sample_data = sensor_data[-1, :, :]
        main_timesteps = sample_data.shape[0]
        n_attention_heads = attn_mean.shape[0]
        attn_timesteps = attn_mean.shape[1]

        # --- MAIN PLOT WITH LEGEND ---
        # Use a sophisticated color palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(cleaned_feature_names)))

        # Plot each feature with enhanced styling
        for i, (feature_name, color) in enumerate(zip(cleaned_feature_names, colors)):
            ax_main.plot(
                sample_data[:, i],
                color=color,
                linewidth=2.5,
                alpha=0.85,
                label=feature_name,
                marker="o",
                markersize=3,
                markevery=max(1, main_timesteps // 20),
            )  # Smart marker placement

        # Style main plot
        ax_main.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
        ax_main.set_ylabel("Feature Value", fontsize=12, fontweight="bold", labelpad=10)
        ax_main.grid(True, alpha=0.2, linestyle="--", linewidth=0.8, color="gray")
        ax_main.set_axisbelow(True)

        # Remove top and right spines for cleaner look
        ax_main.spines["top"].set_visible(False)
        ax_main.spines["right"].set_visible(False)
        ax_main.spines["left"].set_linewidth(1.2)
        ax_main.spines["bottom"].set_linewidth(1.2)
        ax_main.spines["left"].set_color("#333333")
        ax_main.spines["bottom"].set_color("#333333")

        if annot_timesteps and (self.machine_part == "COMPLETE"):
            annot_labels = [
                "Start-Clamping",
                "Start-Bending",
                "Start-Declamping",
                "End-Clamping",
            ]  # Optional short labels

            for ts, label in zip(annot_timesteps, annot_labels):
                # Vertical line for visibility
                ax_main.axvline(
                    ts, color="black", linestyle="--", linewidth=1.2, alpha=0.7
                )

                # Annotated text placed slightly above the data region
                ax_main.annotate(
                    label,
                    xy=(ts, sample_data[:, :].max()),  # anchor at top of plot
                    xytext=(0, 10),  # offset upward
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8),
                )

        ax_main.set_xlim(0, main_timesteps - 1)
        ax_main.set_facecolor("#f9f9f9")
        ax_main.set_title(
            "Sensor Data Over Time", fontsize=14, fontweight="bold", pad=15
        )

        # Add legend with enhanced styling
        legend = ax_main.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=10,
            framealpha=0.95,
            edgecolor="#cccccc",
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_linewidth(1.2)

        # --- ATTENTION HEATMAP ---
        # Ensure heatmap has exactly the same number of timesteps as main plot
        if attn_timesteps != main_timesteps:
            print(
                f"Resizing attention from {attn_timesteps} to {main_timesteps} timesteps"
            )
            attn_data_resized = np.zeros((n_attention_heads, main_timesteps))
            for i in range(n_attention_heads):
                x_original = np.arange(attn_timesteps)
                x_target = np.linspace(0, attn_timesteps - 1, main_timesteps)
                attn_data_resized[i] = np.interp(x_target, x_original, attn_mean[i])
            attn_data = attn_data_resized
        else:
            attn_data = attn_mean

        # Create enhanced heatmap
        im = ax_heatmap.imshow(
            attn_data,
            aspect="auto",
            cmap="magma",  # More visually appealing colormap
            interpolation="bilinear",
            extent=[0, main_timesteps - 1, 0, n_attention_heads - 1],
        )

        # Style heatmap
        ax_heatmap.set_xlabel("Time Step", fontsize=12, fontweight="bold", labelpad=10)
        ax_heatmap.set_ylabel(
            "Attention Head", fontsize=9, fontweight="bold", labelpad=10
        )

        ax_heatmap.set_yticks(np.arange(n_attention_heads))
        # Reverse the label order
        ax_heatmap.set_yticklabels(
            [f"{i + 1}" for i in reversed(range(n_attention_heads))], fontsize=5
        )

        ax_heatmap.set_xlim(0, main_timesteps - 1)
        ax_heatmap.set_facecolor("white")
        ax_heatmap.set_title(
            "Attention Head Intensity", fontsize=14, fontweight="bold", pad=15
        )

        # Remove spines for cleaner look
        ax_heatmap.spines["top"].set_visible(False)
        ax_heatmap.spines["right"].set_visible(False)
        ax_heatmap.spines["left"].set_linewidth(1.2)
        ax_heatmap.spines["bottom"].set_linewidth(1.2)
        ax_heatmap.spines["left"].set_color("#333333")
        ax_heatmap.spines["bottom"].set_color("#333333")

        # Add colorbar with enhanced styling
        cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.9, pad=0.02)
        cbar.set_label("Attention Weight", fontsize=11, fontweight="bold", labelpad=10)
        cbar.ax.tick_params(labelsize=9)
        cbar.outline.set_linewidth(1.2)

        # Fine-tune layout
        plt.tight_layout()

        # Get positions for alignment
        pos_main = ax_main.get_position()
        pos_heat = ax_heatmap.get_position()

        # Make heatmap width match main plot width
        ax_heatmap.set_position(
            [pos_heat.x0, pos_heat.y0, pos_main.width, pos_heat.height]
        )

        fig.savefig(attn_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Reposition colorbar to align properly
        cbar.ax.set_position(
            [pos_main.x0 + pos_main.width + 0.02, pos_heat.y0, 0.015, pos_heat.height]
        )

        fig.savefig(attn_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
