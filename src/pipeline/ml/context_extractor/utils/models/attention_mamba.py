"""
CPU Mamba equivalent of AttentionLSTM with:
- MLPAttention (per-horizon attention)
- optional scalar input fusion
- Mamba encoder (pure PyTorch via mambapy) instead of LSTM

Install:
    pip install mambapy
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mambapy.mamba import Mamba as MambaPy
from mambapy.mamba import MambaConfig as MambaPyConfig


class MLPAttention(nn.Module):
    """
    Same as your MLPAttention:
    Each prediction step has its own learnable embedding and an MLP for scores.
    """
    def __init__(self, n_predictions: int, hidden_dim: int = 128):
        super().__init__()
        self.n_predictions = n_predictions
        self.angle_embeddings = nn.Parameter(torch.randn(n_predictions, hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.xavier_uniform_(self.angle_embeddings)

    def forward(self, H: torch.Tensor):
        B, T, D = H.shape
        contexts, attns = [], []
        for a in range(self.n_predictions):
            scores = self.mlp(H + self.angle_embeddings[a]).squeeze(-1)  # (B,T)
            w = torch.softmax(scores, dim=-1)                            # (B,T)
            ctx = (w.unsqueeze(-1) * H).sum(1)                           # (B,D)
            contexts.append(ctx)
            attns.append(w)
        return torch.stack(contexts, dim=1), torch.stack(attns, dim=1)    # (B,A,D), (B,A,T)


class MambaEncoderCPU(nn.Module):
    """
    CPU Mamba encoder stack.

    Important:
    - mambapy's top-level Mamba model uses n_layers internally.
    - We wrap it to behave like an "encoder" returning (B,T,D).
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        cfg = MambaPyConfig(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
        )
        self.core = MambaPy(cfg)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        y = self.core(x)
        return self.dropout(y)


class AttentionMamba(nn.Module):
    """
    Mamba equivalent to AttentionLSTM:
    - input projection -> Mamba encoder -> LN -> MLPAttention -> optional scalar fusion -> FC head
    """
    def __init__(
        self,
        input_features: int,
        n_predictions: int,
        output_features: int = 1,
        hidden_dim: int = 128,
        mamba_layers: int = 2,
        d_state: int = 16,
        dropout: float = 0.3,
        scalar_embedding_dim: int = 16,
        use_scalar: bool = False,
    ):
        super().__init__()

        self.use_scalar = use_scalar
        self.n_predictions = n_predictions
        self.hidden_dim = hidden_dim

        # Scalar embedding (only if enabled) — identical behavior to your LSTM model
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

        # Project input features to hidden_dim (LSTM implicitly does this; Mamba needs it explicit)
        self.in_proj = nn.Linear(input_features, hidden_dim)

        # Mamba encoder (CPU)
        self.mamba = MambaEncoderCPU(
            d_model=hidden_dim,
            n_layers=mamba_layers,
            d_state=d_state,
            dropout=dropout,
        )

        # Same normalization + attention as your original module
        self.ln = nn.LayerNorm(hidden_dim)
        self.attention = MLPAttention(n_predictions, hidden_dim)

        # Same style of prediction head
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
            x: (B, T, input_features)
            scalar: (B,) or (B,1) – ignored if use_scalar=False
            config: (B, config_dim) – currently unused
        Returns:
            out:  (B, n_predictions, output_features)
            attn: (B, n_predictions, T)
        """
        if x.is_cuda:
            raise RuntimeError("This AttentionMambaCPU is CPU-only. Move inputs/model to CPU.")

        # Encode
        h = self.in_proj(x)     # (B,T,H)
        h = self.mamba(h)       # (B,T,H)
        h = self.ln(h)          # (B,T,H)

        # Attention pooling (per prediction horizon)
        ctx, attn = self.attention(h)  # (B,A,H), (B,A,T)

        # Optional scalar fusion (same as your original)
        if self.use_scalar:
            if scalar is None:
                raise ValueError("scalar input is required when use_scalar=True")
            if scalar.dim() == 1:
                scalar = scalar.unsqueeze(-1)  # (B,1)

            scalar_emb = self.scalar_embedding(scalar)                 # (B,S)
            scalar_emb = scalar_emb.unsqueeze(1).expand(-1, self.n_predictions, -1)  # (B,A,S)
            combined = torch.cat([ctx, scalar_emb], dim=-1)            # (B,A,H+S)
        else:
            combined = ctx

        out = self.fc(combined)  # (B,A,output_features)
        return out, attn
