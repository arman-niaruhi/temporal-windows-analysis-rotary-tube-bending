import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# True vs Predicted (index-based line plot)
# --------------------------------------------------
def plot_predictions(y_true, y_pred, save_path: str = None, show: bool = False):
    plt.figure(figsize=(12, 5))

    plt.plot(
        y_true,
        label="True",
        linewidth=2.5,
        alpha=0.9
    )
    plt.plot(
        y_pred,
        label="Predicted",
        linewidth=2,
        linestyle="--",
        alpha=0.8
    )

    plt.title("Predicted vs True Springback")
    plt.xlabel("Sample Index")
    plt.ylabel("Springback")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()


# --------------------------------------------------
# True vs Predicted (scatter with ideal line)
# --------------------------------------------------
def plot_true_vs_pred(y_true, y_pred, r2, save_path: str = None, show: bool = False):
    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.5,
        s=40,
        edgecolor="none"
    )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2
    )

    plt.xlabel("True Springback")
    plt.ylabel("Predicted Springback")
    plt.title(f"True vs Predicted (R² = {r2:.4f})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()


# --------------------------------------------------
# Residual diagnostics
# --------------------------------------------------
def plot_residuals(y_true, y_pred, save_path: str = None, show: bool = False):
    residuals = y_pred - y_true

    # --- Residuals vs Predicted ---
    plt.figure(figsize=(7, 5))
    plt.scatter(
        y_pred,
        residuals,
        alpha=0.5,
        s=40,
        edgecolor="none"
    )
    plt.axhline(0, linestyle="--", linewidth=2)

    plt.xlabel("Predicted Springback")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()

    # --- Residual Distribution ---
    plt.figure(figsize=(7, 5))
    plt.hist(
        residuals,
        bins=40,
        alpha=0.8
    )
    plt.axvline(0, linestyle="--", linewidth=2)

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()


# --------------------------------------------------
# Training history plots
# --------------------------------------------------
def plot_training_history(history, save_path: str = None, show: bool = False):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # --- Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train", linewidth=2.5)
    plt.plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()

    # --- R² ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_r2"], label="Train R²", linewidth=2.5)
    plt.plot(epochs, history["val_r2"], label="Validation R²", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("R² History")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path.replace(".png", "_r2.png"), dpi=150)

    if show:
        plt.show()

    plt.close()

    # --- Learning Rate ---
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["lr"], linewidth=2.5)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()
