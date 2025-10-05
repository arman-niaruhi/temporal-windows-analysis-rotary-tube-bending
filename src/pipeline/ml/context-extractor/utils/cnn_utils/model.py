import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import joblib
from tqdm import tqdm


class MultiOutputCNN(nn.Module):
    def __init__(
        self, input_size=17, seq_len=88, output_size=4, num_angles=5, hidden_dim=128
    ):
        super().__init__()
        self.num_angles = num_angles
        self.seq_len = seq_len

        # CNN for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
        )

        # Separate output heads for each angle
        self.angle_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, output_size),
                )
                for _ in range(num_angles)
            ]
        )

        # Angle embeddings
        self.angle_emb = nn.Embedding(num_angles, 128)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.transpose(1, 2)

        # Extract features
        features = self.conv_layers(x).squeeze(-1)  # (batch, 128)

        # Generate outputs for each angle
        outputs = []
        for angle_idx in range(self.num_angles):
            angle_emb = self.angle_emb(torch.tensor(angle_idx, device=x.device))
            # Combine features with angle information
            combined = features + angle_emb.unsqueeze(0).expand(features.size(0), -1)
            output = self.angle_heads[angle_idx](combined)
            outputs.append(output)

        return torch.stack(outputs, dim=1)  # (batch, num_angles, output_size)

    def train_cnn_model(self, train_loader, test_loader, num_epochs=100, lr=0.001, save_path=None):
        """
        Train CNN model with batch-level tqdm progress and optionally save.
        """
        device = next(self.parameters()).device  # detect model device
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        train_losses = []
        test_losses = []

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0
            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, colour="green", total=len(train_loader)):
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = self(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))

            # Validation
            self.eval()
            test_loss = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    output = self(xb)
                    loss = criterion(output, yb)
                    test_loss += loss.item()
            test_losses.append(test_loss / len(test_loader))

            # Update outer tqdm description in-place
            tqdm.write(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}")

        # Save model safely
        if save_path:
            torch.save(self.state_dict(), save_path)
            print(f"Model saved to: {save_path}")

        return self, train_losses, test_losses

    def visualize_cnn_importance(self, test_dataset, sample_idx=0):
        """
        Visualize which input features are most important using gradient-based methods
        """
        self.eval()

        # Get sample
        x_sample, y_true = test_dataset[sample_idx]
        x_sample = x_sample.unsqueeze(0).requires_grad_(True)

        # Compute gradients for each angle
        importance_maps = []

        for angle_idx in range(self.num_angles):
            # Forward pass for specific angle
            output = self(x_sample)
            target_output = output[0, angle_idx].sum()  # Sum all output dimensions

            # Backward pass to get gradients
            self.zero_grad()
            target_output.backward()

            # Get gradients w.r.t input
            gradients = x_sample.grad.data.abs().squeeze(0)  # (seq_len, input_size)
            importance_maps.append(gradients.numpy())

        importance_maps = np.array(importance_maps)  # (num_angles, seq_len, input_size)

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for angle_idx in range(5):
            importance = importance_maps[angle_idx]

            # Plot heatmap of important features over time
            im = axes[angle_idx].imshow(importance.T, aspect="auto", cmap="hot")
            axes[angle_idx].set_title(f"Angle {angle_idx} - Feature Importance")
            axes[angle_idx].set_xlabel("Time Step")
            axes[angle_idx].set_ylabel("Feature Index")
            plt.colorbar(im, ax=axes[angle_idx])

        # Plot overall feature importance
        overall_importance = importance_maps.mean(
            axis=(0, 1)
        )  # Average over angles and time
        axes[5].bar(range(len(overall_importance)), overall_importance)
        axes[5].set_title("Overall Feature Importance")
        axes[5].set_xlabel("Feature Index")
        axes[5].set_ylabel("Average Gradient Magnitude")

        plt.tight_layout()
        plt.show()

        return importance_maps
