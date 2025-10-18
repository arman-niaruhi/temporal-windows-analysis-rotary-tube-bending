from utils.lstm_utils.lstm_preprocessing_utils import WindowAlgPreprocessor
from utils.lstm_utils.model import AngleAwareLSTM
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import joblib
import numpy as np
from tqdm import trange

lstm_preprocessor = WindowAlgPreprocessor(sensors_path=f"data/ml/features_machine_and_movement_complete.csv", target_path=f"data/ml/targets_movement_complete.csv")
sensors_df, target_df = lstm_preprocessor.read_data()
sensors_df = lstm_preprocessor.feature_selection()
lstm_preprocessor.normalize_angle()
X = lstm_preprocessor.group_and_pad(lstm_preprocessor.sensor_df, group_col="Experiment_ID")[:,::10,:]
Y = lstm_preprocessor.group_and_pad(lstm_preprocessor.target_df, group_col="Experiment_ID")[:,:-2:9,1:]
X = lstm_preprocessor.patchifier(X=X, window_size=10)


def train_model(
    model, X_train, Y_train, angles_train,
    X_val, Y_val, angles_val,
    epochs=1000, lr=1e-4, patience=20, device='cpu',
    scheduler_type='ReduceLROnPlateau', min_lr=1e-6, factor=0.5
):
    model.to(device)
    X_train, Y_train, angles_train = X_train.to(device), Y_train.to(device), angles_train.to(device)
    X_val, Y_val, angles_val = X_val.to(device), Y_val.to(device), angles_val.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Scheduler
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=5, min_lr=min_lr)
    elif scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=factor)
    else:
        scheduler = None

    best_val_loss = float('inf')
    patience_counter = 5
    best_model_state = None
    # create the iterator
    pbar = trange(epochs, desc="Training", unit="epoch")

    for epoch in pbar:
        # -----------------
        # Training
        # -----------------
        model.train()
        optimizer.zero_grad()
        preds = model(X_train, angles_train)
        loss = loss_fn(preds, Y_train)
        loss.backward()
        optimizer.step()

        # -----------------
        # Validation
        # -----------------
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val, angles_val)
            val_loss = loss_fn(val_preds, Y_val).item()

        # -----------------
        # Scheduler step
        # -----------------
        if scheduler_type == 'ReduceLROnPlateau' and scheduler is not None:
            scheduler.step(val_loss)
        elif scheduler_type == 'StepLR' and scheduler is not None:
            scheduler.step()

        # -----------------
        # Early stopping
        # -----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Current LR
        current_lr = optimizer.param_groups[0]['lr']

        # Print info in tqdm
        pbar.write(f"Epoch {epoch:03d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6e}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.6f}")
            break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss


# -----------------------------
# Flatten dataset per angle
# -----------------------------
num_samples, seq_len, num_features = X.shape
num_angles, num_outputs = Y.shape[1:]

X_flat, Y_flat, angle_flat = [], [], []
for i in range(num_samples):
    for j in range(num_angles):
        X_flat.append(X[i])
        Y_flat.append(Y[i, j])
        angle_flat.append([j / (num_angles - 1)])

X_flat = torch.tensor(np.array(X_flat), dtype=torch.float32)
Y_flat = torch.tensor(np.array(Y_flat), dtype=torch.float32)
angle_flat = torch.tensor(np.array(angle_flat), dtype=torch.float32)

# -----------------------------
# Split into train/validation
# -----------------------------
X_train, X_val, Y_train, Y_val, angles_train, angles_val = train_test_split(
    X_flat, Y_flat, angle_flat, test_size=0.2, random_state=42
)

# -----------------------------
# Initialize model and train
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AngleAwareLSTM(input_dim=num_features, output_dim=num_outputs, bidirectional=True)

model, best_val_loss = train_model(
    model, X_train, Y_train, angles_train,
    X_val, Y_val, angles_val,
    epochs=500, lr=1e-4, patience=20, device=device,
    scheduler_type='ReduceLROnPlateau'
)


# Save the best model
joblib.dump(model, f"models/lstm/bi_lstm_movement_machine_{num_angles}_angle.joblib")
print(f"Model saved. Best validation loss: {best_val_loss:.6f}")
