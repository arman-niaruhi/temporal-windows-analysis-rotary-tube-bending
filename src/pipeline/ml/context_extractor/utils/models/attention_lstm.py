"""
Methods Module: Attention LSTM

This module implements:
- AttentionLSTM model with MLPAttention

References:
------------
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
- Vaswani, A., et al. (2017). Attention is all you need. Advances in neural information processing systems, 30.
- Choi, E., et al. (2016). RETAIN: An interpretable predictive model for healthcare using reverse time attention mechanism. 
"""
import torch
import torch.nn as nn


class MLPAttention(nn.Module):
    """
    Implements a multi-layer perceptron (MLP) attention mechanism.
    
    Each prediction timestep has its own learnable embedding (angle embedding),
    which allows the model to focus on different parts of the input sequence
    for each prediction.
    """
    def __init__(self, n_predictions: int, hidden_dim: int = 128):
        """
        Initialize MLPAttention.

        Args:
            n_predictions: Number of prediction timesteps
            hidden_dim: Hidden dimensionality of input features
        """
        super().__init__()
        self.n_predictions = n_predictions

        # Learnable embeddings for each prediction step
        self.angle_embeddings = nn.Parameter(torch.randn(n_predictions, hidden_dim))

        # Small MLP to compute attention scores
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Outputs scalar attention score per timestep
        )

        # Xavier initialization for angle embeddings
        nn.init.xavier_uniform_(self.angle_embeddings)

    def forward(self, H: torch.Tensor):
        """
        Forward pass for attention.

        Args:
            H: Input tensor of shape (batch_size, sequence_length, hidden_dim)

        Returns:
            contexts: Context vectors of shape (batch_size, n_predictions, hidden_dim)
            attns: Attention weights of shape (batch_size, n_predictions, sequence_length)
        """
        B, T, D = H.shape
        contexts, attns = [], []

        # Compute attention for each prediction timestep
        for a in range(self.n_predictions):
            # Attention scores: MLP(H + angle_embedding)
            scores = self.mlp(H + self.angle_embeddings[a]).squeeze(-1)
            # Normalize scores to probabilities
            w = torch.softmax(scores, dim=-1)
            # Weighted sum of input features to obtain context vector
            ctx = (w.unsqueeze(-1) * H).sum(1)
            contexts.append(ctx)
            attns.append(w)

        # Stack contexts and attention weights across prediction timesteps
        return torch.stack(contexts, dim=1), torch.stack(attns, dim=1)



class AttentionLSTM(nn.Module):
    """
    LSTM model with MLP-based attention for multistep sequence prediction.

    Architecture:
        1. LSTM encoder to extract temporal features
        2. Layer normalization for stabilized hidden states
        3. MLPAttention for interpretable attention per prediction timestep
        4. Fully connected layers to map context vectors to predictions
    """
    def __init__(
        self,
        input_features: int,
        n_predictions: int,
        output_features: int = 1,
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize AttentionLSTM.

        Args:
            input_features: Number of input features per timestep
            n_predictions: Number of timesteps to predict
            output_features: Number of output features per timestep
            hidden_dim: Hidden dimension of LSTM and attention layers
            lstm_layers: Number of stacked LSTM layers
            dropout: Dropout rate applied to LSTM layers (except single layer)
        """
        super().__init__()
        self.input_features = input_features
        self.n_predictions = n_predictions
        self.output_features = output_features
        self.hidden_dim = hidden_dim

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_features,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Layer normalization for LSTM outputs
        self.ln = nn.LayerNorm(hidden_dim)

        # Attention mechanism
        self.attention = MLPAttention(n_predictions, hidden_dim)

        # Fully connected layers to generate final predictions
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_features),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of AttentionLSTM.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)

        Returns:
            out: Predictions of shape (batch_size, n_predictions, output_features)
            attn: Attention weights of shape (batch_size, n_predictions, sequence_length)
        """
        # Encode input sequence
        o, _ = self.lstm(x)
        # Normalize LSTM outputs
        o = self.ln(o)
        # Apply attention to LSTM outputs
        ctx, attn = self.attention(o)
        # Map context vectors to predictions
        out = self.fc(ctx)
        return out, attn
