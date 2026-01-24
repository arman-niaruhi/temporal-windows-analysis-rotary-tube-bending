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
        use_scalar: bool = True,   # <<< FLAG
        config_dim: int | None = None,
        config_embedding_dim: int = 16,
        use_config: bool = True,
        split_output_heads: bool = False,
        main_head_hidden_sizes: list[int] | None = None,
        secondary_head_hidden_sizes: list[int] | None = None,
    ):
        super().__init__()
        print(use_scalar, use_config)
        self.use_scalar = use_scalar
        self.use_config = use_config
        self.n_predictions = n_predictions
        self.hidden_dim = hidden_dim
        self.output_features = output_features
        self.split_output_heads = split_output_heads

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

        def _build_mlp_head(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = input_dim
            for size in hidden_sizes:
                layers.append(nn.Linear(prev, size))
                layers.append(nn.ReLU())
                prev = size
            layers.append(nn.Linear(prev, output_dim))
            return nn.Sequential(*layers)

        # Final prediction head(s)
        if self.split_output_heads and output_features > 1:
            base_main_sizes = main_head_hidden_sizes or [
                max(1, hidden_dim // 2),
                max(1, hidden_dim // 4),
            ]
            base_secondary_sizes = secondary_head_hidden_sizes or [
                max(1, hidden_dim // 2),
                max(1, hidden_dim // 4),
                max(1, hidden_dim // 8),
            ]
            self.fc_heads = nn.ModuleList()
            for i in range(output_features):
                sizes = base_secondary_sizes if i == 0 else base_main_sizes
                self.fc_heads.append(_build_mlp_head(combined_dim, sizes, 1))
            self.fc = None
        else:
            self.fc = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, output_features),
            )
            self.fc_heads = None

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

        if self.fc_heads is not None:
            outs = [head(combined) for head in self.fc_heads]
            out = torch.cat(outs, dim=-1)
        else:
            out = self.fc(combined)
        return out, attn
