import json
import torch
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

from src.pipeline.ml.context_extractor.utils.lstm_utils.config.seed import enforce_reproducibility
from src.pipeline.ml.context_extractor.utils.lstm_utils.data.data_preprocessor import prepare_data
from src.pipeline.ml.context_extractor.utils.lstm_utils.models.att_lstm import AttentionLSTM


def load_model_from_mlflow(run_id=None, model_path=None, device='cpu'):
    """Load trained model from MLflow."""
    if run_id:
        model_uri = f"runs:/{run_id}/model"
    elif model_path:
        model_uri = model_path
    else:
        raise ValueError("Either run_id or model_path must be provided")
    
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.eval()
    model.to(device)
    print(f"Model loaded from: {model_uri}")
    print(f"Model device: {device}")
    return model


def main():
    # ------------------------  
    # Config & Data  
    # ------------------------
    with open("src/pipeline/ml/context_extractor/utils/lstm_utils/config/lstm_config.json", "r") as f:
        config = json.load(f)
    
    enforce_reproducibility(seed=config.get("seed", 42))
    input_path_param = config.get("input_path_param")
    preprocessing_param = config.get("preprocessing_param")

    X, Y, sensor_names, target_feature_names, annot_timesteps, mandrel_extraction_annot_timesteps = prepare_data(
        input_path_param=input_path_param,
        preprocessing_param=preprocessing_param
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_EXPERIMENTS, TIMESTEPS, FEATURES_IN = X.shape
    _, PRED_TIMESTEPS, FEATURES_OUT = Y.shape
    param = config.get("training_param")

    # ------------------------
    # Model
    # ------------------------
    model = AttentionLSTM(
        input_features=FEATURES_IN,
        n_predictions=PRED_TIMESTEPS,
        output_features=FEATURES_OUT,
        hidden_dim=param["hidden_dim"],
        lstm_layers=param["lstm_layers"],
        dropout=param["dropout"],
    ).to(device)

    model = load_model_from_mlflow(
        run_id="c87ebf15515b4f04bc5f6bbbb0225623",
        device=device
    )
    model.eval()

    # ------------------------
    # Interactive Plot
    # ------------------------
    sample_idx = 0  # Python 0-indexed
    angle_idx = 0

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    num_angles = Y.shape[1]
    num_features = Y.shape[2]
    input_features = X.shape[2]

    def update_plot():
        for ax in axs.flatten():
            ax.clear()

        out = model(X[sample_idx:sample_idx+1])

        # 1. All target features (model output)
        for f in range(num_features):
            target_curve = out[0][0, :, f].detach().cpu().numpy().flatten()
            axs[0, 0].plot(target_curve, label=f"Feature {f}", alpha=0.7)
        axs[0, 0].set_title(f"All Target Features (Model Output) | Sample {sample_idx+1}")
        axs[0, 0].set_xlabel("Angle Index")
        axs[0, 0].set_ylabel("Value")
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend(loc='upper right', fontsize='small')

        # 2. True target for selected sample (from Y)
        true_target = Y[sample_idx, :, :].detach().cpu().numpy()
        for f in range(num_features):
            axs[0, 1].plot(true_target[:, f], label=f"Feature {f}", linewidth=2)
        axs[0, 1].set_title(f"True Target | Sample {sample_idx+1}")
        axs[0, 1].set_xlabel("Angle Index")
        axs[0, 1].set_ylabel("Value")
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend(loc='upper right', fontsize='small')

        # 3. Predicted (attention-weighted) for selected angle
        pred_curve = out[1][0, angle_idx, :].detach().cpu().numpy().flatten()
        axs[1, 0].plot(pred_curve, linewidth=2, color='red', label="Predicted (Attention)")

        # Optional attention weights
        if len(out) > 2:
            attn_weights = out[2][0, angle_idx, :].detach().cpu().numpy().flatten()
            axs_att = axs[1, 0].twinx()
            axs_att.plot(attn_weights, color='blue', linestyle='--', alpha=0.5, label="Attention")
            axs_att.set_ylabel("Attention Weight")
            axs_att.set_ylim(0, 1)
            axs_att.legend(loc='upper left', fontsize='small')

        axs[1, 0].set_title(f"Predicted | Sample {sample_idx+1}, Angle {angle_idx}")
        axs[1, 0].set_xlabel("Feature Index")
        axs[1, 0].set_ylabel("Value")
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend(loc='upper right', fontsize='small')

        # 4. Input features
        input_curve = X[sample_idx, :, :].detach().cpu().numpy()
        for i in range(input_features):
            axs[1, 1].plot(input_curve[:, i], label=f"Input Feature {i}", alpha=0.7)
        axs[1, 1].set_title(f"Input Data | Sample {sample_idx+1}")
        axs[1, 1].set_xlabel("Angle Index")
        axs[1, 1].set_ylabel("Value")
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].legend(loc='upper right', fontsize='small')

        fig.canvas.draw_idle()

    # Initial draw
    update_plot()

    # ------------------------
    # Keyboard navigation
    # ------------------------
    def on_key(event):
        nonlocal sample_idx, angle_idx
        if event.key == 'left':
            sample_idx = (sample_idx - 1) % X.shape[0]   # wrap around
        elif event.key == 'right':
            sample_idx = (sample_idx + 1) % X.shape[0]
        elif event.key == 'up':
            angle_idx = (angle_idx + 1) % num_angles
        elif event.key == 'down':
            angle_idx = (angle_idx - 1) % num_angles
        update_plot()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


if __name__ == "__main__":
    main()
