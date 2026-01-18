import matplotlib.pyplot as plt


def plot_predictions(y_true, y_pred):
    plt.figure()
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.title("Predicted vs. True Springback")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_true_vs_pred(y_true, y_pred, r2):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"True vs Predicted (R² = {r2:.4f})")
    plt.grid(True)
    plt.show()


def plot_residuals(y_true, y_pred):
    residuals = y_pred - y_true

    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.hist(residuals, bins=40)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.grid(True)
    plt.show()


def plot_training_history(history):
    plt.figure()
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(history["train_r2"], label="Train R²")
    plt.plot(history["val_r2"], label="Val R²")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("R² History")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(history["lr"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.show()
