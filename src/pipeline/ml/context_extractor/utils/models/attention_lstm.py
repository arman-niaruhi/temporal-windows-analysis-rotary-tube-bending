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
    def __init__(self, n_predictions: int, hidden_dim: int = 128):
        super().__init__()
        self.n_predictions = n_predictions
        self.angle_embeddings = nn.Parameter(torch.randn(n_predictions, hidden_dim))
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.angle_embeddings)

    def forward(self, H: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, D = H.shape

        H_proj = self.W_h(H)                       # (B,T,D)
        Q_proj = self.W_q(self.angle_embeddings)    # (P,D)

        scores = self.v(
            torch.tanh(H_proj.unsqueeze(1) + Q_proj.unsqueeze(0).unsqueeze(2))
        ).squeeze(-1)                               # (B,P,T)

        mask_bool = None
        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool, device=scores.device).unsqueeze(1)  # (B,1,T)
            scores = scores.masked_fill(~mask_bool, float("-inf"))

        w = torch.softmax(scores, dim=-1)           # (B,P,T)

        # Robustness: if a row was all -inf (shouldn't happen if lengths>=1), softmax gives NaN.
        # Replace NaNs with zeros.
        w = torch.nan_to_num(w, nan=0.0)

        if mask_bool is not None:
            # Renormalize so weights sum to 1 over valid timesteps
            w = w * mask_bool
            denom = w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            w = w / denom

        ctx = (w.unsqueeze(-1) * H.unsqueeze(1)).sum(dim=2)  # (B,P,D)
        return ctx, w



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
        use_feature_attention: bool = False,
        use_angle_embedding: bool = False,
        angle_embedding_dim: int = 8,
    ):
        super().__init__()
        print(use_scalar, use_config)
        self.use_scalar = use_scalar
        self.use_config = use_config
        self.n_predictions = n_predictions
        self.hidden_dim = hidden_dim
        self.output_features = output_features
        self.split_output_heads = split_output_heads
        self.use_feature_attention = bool(use_feature_attention) and output_features > 1
        self.use_angle_embedding = use_angle_embedding

        if self.use_feature_attention and not self.split_output_heads:
            raise ValueError(
                "use_feature_attention=True requires split_output_heads=True when output_features > 1."
            )

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

        if self.use_angle_embedding:
            if angle_embedding_dim <= 0:
                raise ValueError("angle_embedding_dim must be > 0 when use_angle_embedding=True")
            self.angle_embedding = nn.Embedding(n_predictions, angle_embedding_dim)
            combined_dim = combined_dim + angle_embedding_dim
        else:
            self.angle_embedding = None

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_features,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.ln = nn.LayerNorm(hidden_dim)
        if self.use_feature_attention:
            self.feature_attentions = nn.ModuleList(
                [MLPAttention(n_predictions, hidden_dim) for _ in range(output_features)]
            )
            self.attention = None
        else:
            self.feature_attentions = None
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
        mask: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (batch_size, seq_len, input_features)
            scalar: (batch_size, 1) or (batch_size,) – ignored if use_scalar=False
            config: (batch_size, config_dim) – ignored if use_config=False
        """
        attention_mask = mask
        if attention_mask is None:
            # Treat all-zero timesteps as padding.
            attention_mask = x.abs().sum(dim=-1) > 0

        # LSTM encoding
        o, _ = self.lstm(x)
        o = self.ln(o)

        # Attention
        if self.use_feature_attention:
            ctx_list = []
            attn_list = []
            for attn_module in self.feature_attentions:
                ctx_f, attn_f = attn_module(o, mask=attention_mask)
                ctx_list.append(ctx_f)
                attn_list.append(attn_f)
            ctx = torch.stack(ctx_list, dim=1)  # (B, F, n_predictions, hidden_dim)
            attn = torch.stack(attn_list, dim=1)  # (B, F, n_predictions, seq_len)
        else:
            ctx, attn = self.attention(o, mask=attention_mask)  # (B, n_predictions, hidden_dim)

        use_scalar = getattr(self, "use_scalar", False)
        use_config = getattr(self, "use_config", False)

        scalar_emb = None
        config_emb = None
        angle_emb = None

        if use_scalar:
            if scalar is None:
                raise ValueError("scalar input is required when use_scalar=True")
            if scalar.dim() == 1:
                scalar = scalar.unsqueeze(-1)
            scalar_emb = self.scalar_embedding(scalar)
            scalar_emb = scalar_emb.unsqueeze(1).expand(-1, self.n_predictions, -1)

        if use_config:
            if config is None:
                raise ValueError("config input is required when use_config=True")
            if config.dim() == 1:
                config = config.unsqueeze(-1)
            config_emb = self.config_embedding(config)
            config_emb = config_emb.unsqueeze(1).expand(-1, self.n_predictions, -1)

        if self.use_angle_embedding:
            angle_idx = torch.arange(self.n_predictions, device=o.device)
            angle_emb = self.angle_embedding(angle_idx).unsqueeze(0).expand(o.size(0), -1, -1)

        if self.use_feature_attention:
            outs = []
            for i, head in enumerate(self.fc_heads):
                combined = ctx[:, i, :, :]
                if scalar_emb is not None:
                    combined = torch.cat([combined, scalar_emb], dim=-1)
                if config_emb is not None:
                    combined = torch.cat([combined, config_emb], dim=-1)
                if angle_emb is not None:
                    combined = torch.cat([combined, angle_emb], dim=-1)
                outs.append(head(combined))
            out = torch.cat(outs, dim=-1)
        else:
            combined = ctx
            if scalar_emb is not None:
                combined = torch.cat([combined, scalar_emb], dim=-1)
            if config_emb is not None:
                combined = torch.cat([combined, config_emb], dim=-1)
            if angle_emb is not None:
                combined = torch.cat([combined, angle_emb], dim=-1)
            if self.fc_heads is not None:
                outs = [head(combined) for head in self.fc_heads]
                out = torch.cat(outs, dim=-1)
            else:
                out = self.fc(combined)
        return out, attn
