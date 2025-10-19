import torch
import joblib
from src.pipeline.ml.classification.utils.model import LSTMClassifier
import numpy as np
from scipy.ndimage import binary_opening, binary_closing


def predict_activity(sensors_df, model_path):
    feature_cols = sensors_df.columns.difference(["Experiment_ID", "Label"])
    input_size = len(feature_cols)
    hidden_size = 64
    num_layers = 2
    num_classes = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        input_size, hidden_size, num_layers, num_classes, bidirectonal=True
    ).to(device)

    # load state_dict via joblib
    state_dict = joblib.load(f"{model_path}/Activity_Detector.joblib")
    model.load_state_dict(state_dict)
    model.eval()

    X = torch.tensor(sensors_df[feature_cols].values, dtype=torch.float32).to(device)
    X = X.unsqueeze(0)  # [1, seq_len, num_features]

    # --- Model predictions ---
    with torch.no_grad():
        outputs = model(X)
        y_pred = torch.argmax(outputs, dim=-1).squeeze(0).cpu().numpy()

    label_mapping = {
        0: "Bending",
        1: "Clamping",
        2: "Declamping",
        3: "Mandrel Extraction",
        4: "Idle",
    }
    
    # Size of neighborhood to consider
    kernel_size = 10  # adjust: larger = more aggressive smoothing

    # Process each unique label
    smoothed = np.zeros_like(y_pred)
    for label in np.unique(y_pred):
        mask = y_pred == label
        # Apply morphological closing (fills small gaps)
        mask = binary_closing(mask, structure=np.ones(kernel_size))
        # Apply morphological opening (removes small noise)
        mask = binary_opening(mask, structure=np.ones(kernel_size))
        smoothed[mask] = label
   
    # y_pred = smoothed
    Y_pred_label = [label_mapping.get(label_idx) for label_idx in y_pred]
    return Y_pred_label
