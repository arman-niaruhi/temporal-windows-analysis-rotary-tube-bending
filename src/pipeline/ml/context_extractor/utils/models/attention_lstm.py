"""
Methods Module: Attention LSTM with Scalar Input

This module implements:
- AttentionLSTM model with MLPAttention and scalar input support

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
    def __init__(
        self,
        input_features: int,
        n_predictions: int,
        output_features: int = 1,
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        scalar_embedding_dim: int = 16,
        use_scalar: bool = False,   # <<< FLAG
        config_dim: int | None = None,
        config_embedding_dim: int = 16,
        use_config: bool = False,
    ):
        super().__init__()

        self.use_scalar = use_scalar
        self.use_config = use_config
        self.n_predictions = n_predictions
        self.hidden_dim = hidden_dim

        # Scalar embedding (only if enabled)
        if self.use_scalar:
            self.scalar_embedding = nn.Sequential(
                nn.Linear(1, scalar_embedding_dim),
                nn.ReLU(),
                nn.Linear(scalar_embedding_dim, scalar_embedding_dim),
            )
            combined_dim = hidden_dim + scalar_embedding_dim
        else:
            self.scalar_embedding = None
            combined_dim = hidden_dim

        # Experiment configuration embedding (only if enabled)
        if self.use_config:
            if config_dim is None or config_dim <= 0:
                raise ValueError("config_dim must be set when use_config=True")
            self.config_embedding = nn.Sequential(
                nn.Linear(config_dim, config_embedding_dim),
                nn.ReLU(),
                nn.Linear(config_embedding_dim, config_embedding_dim),
            )
            combined_dim = combined_dim + config_embedding_dim
        else:
            self.config_embedding = None

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_features,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.ln = nn.LayerNorm(hidden_dim)
        self.attention = MLPAttention(n_predictions, hidden_dim)

        # Final prediction head
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_features),
        )

    def forward(
        self,
        x: torch.Tensor,
        scalar: torch.Tensor | None = None,
        config: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (batch_size, seq_len, input_features)
            scalar: (batch_size, 1) or (batch_size,) – ignored if use_scalar=False
            config: (batch_size, config_dim) – ignored if use_config=False
        """
        # LSTM encoding
        o, _ = self.lstm(x)
        o = self.ln(o)

        # Attention
        ctx, attn = self.attention(o)  # (B, n_predictions, hidden_dim)

        combined = ctx

        use_scalar = getattr(self, "use_scalar", False)
        use_config = getattr(self, "use_config", False)

        if use_scalar:
            if scalar is None:
                raise ValueError("scalar input is required when use_scalar=True")

            if scalar.dim() == 1:
                scalar = scalar.unsqueeze(-1)

            scalar_emb = self.scalar_embedding(scalar)
            scalar_emb = scalar_emb.unsqueeze(1).expand(-1, self.n_predictions, -1)

            combined = torch.cat([combined, scalar_emb], dim=-1)

        if use_config:
            if config is None:
                raise ValueError("config input is required when use_config=True")

            if config.dim() == 1:
                config = config.unsqueeze(-1)

            config_emb = self.config_embedding(config)
            config_emb = config_emb.unsqueeze(1).expand(-1, self.n_predictions, -1)

            combined = torch.cat([combined, config_emb], dim=-1)

        out = self.fc(combined)
        return out, attn
