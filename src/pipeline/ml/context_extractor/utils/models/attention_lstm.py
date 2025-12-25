import torch
import torch.nn as nn


class MLPAttention(nn.Module):
    def __init__(self, n_predictions: int, hidden_dim: int = 128):
        """Initialize MLPAttention."""
        super().__init__()
        self.n_predictions = n_predictions
        self.angle_embeddings = nn.Parameter(torch.randn(n_predictions, hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.xavier_uniform_(self.angle_embeddings)

    def forward(self, H):
        """Forward pass for MLPAttention.

        Args:
            H: Input tensor of shape (batch_size, sequence_length, hidden_dim)

        Returns:
            contexts: Context vectors of shape (batch_size, n_predictions, hidden_dim)
            attns: Attention weights of shape (batch_size, n_predictions, sequence_length)
        """
        B, T, D = H.shape
        contexts, attns = [], []
        for a in range(self.n_predictions):
            scores = self.mlp(H + self.angle_embeddings[a]).squeeze(-1)
            w = torch.softmax(scores, dim=-1)
            ctx = (w.unsqueeze(-1) * H).sum(1)
            contexts.append(ctx)
            attns.append(w)
        return torch.stack(contexts, dim=1), torch.stack(attns, dim=1)


class AttentionLSTM(nn.Module):
    def __init__(
        self,
        input_features: int,
        n_predictions: int,
        output_features: int = 1,
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        """Initialize AttentionLSTM model.

        Args:
            input_features: Number of input features.
            n_predictions: Number of predictions to make.
            output_features: Number of output features.
            hidden_dim: Hidden dimension of the LSTM.
            lstm_layers: Number of LSTM layers.
            dropout: Dropout rate for LSTM layers.

        """
        super().__init__()
        self.input_features = input_features
        self.n_predictions = n_predictions
        self.output_features = output_features
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_features,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.attention = MLPAttention(n_predictions, hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_features),
        )

    def forward(self, x):
        o, _ = self.lstm(x)
        o = self.ln(o)
        ctx, attn = self.attention(o)
        out = self.fc(ctx)
        return out, attn
