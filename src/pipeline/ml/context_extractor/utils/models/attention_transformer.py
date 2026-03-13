from __future__ import annotations

import torch
import torch.nn as nn


class StepQueryAttention(nn.Module):
    """
    Step-specific multi-head cross-attention using learnable query embeddings.

    Inputs:
      H:    (B, T, D)   encoded sequence
      mask: (B, T) bool True=valid, False=pad (optional)

    Returns:
      ctx:  (B, A, D)
      attn: (B, A, T)   average over heads
    """

    def __init__(
        self,
        n_predictions: int,
        d_model: int,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_predictions = n_predictions
        self.d_model = d_model
        self.n_heads = n_heads

        # Learnable step queries: (A, D)
        self.step_queries = nn.Parameter(torch.empty(n_predictions, d_model))
        nn.init.xavier_uniform_(self.step_queries)

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, H: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, D = H.shape
        if D != self.d_model:
            raise ValueError("H last dim must match d_model")

        Q = self.step_queries.unsqueeze(0).expand(B, -1, -1)  # (B, A, D)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()  # True = ignore

        ctx, attn_w = self.mha(
            query=Q,
            key=H,
            value=H,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        # (B, n_heads, A, T) -> (B, A, T)
        attn = attn_w.mean(dim=1) if attn_w.dim() == 4 else attn_w
        return ctx, attn


class TransformerAttention(nn.Module):
    """
    Transformer encoder + step-query attention + optional scalar side input.

    Inputs:
      x:      (B, T, input_features)
      scalar: (B,) or (B,1) if use_scalar=True
      mask:   (B, T) bool True=valid, False=pad (optional)

    Returns:
      out:  (B, A, output_features)
      attn: (B, A, T)
    """

    def __init__(
        self,
        input_features: int,
        n_predictions: int,
        output_features: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        transformer_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        scalar_embedding_dim: int = 16,
        use_scalar: bool = False,
        use_layernorm: bool = True,
        max_len: int = 4096,
        attn_dropout: float = 0.0,
        use_angle_embedding: bool = False,
        angle_embedding_dim: int = 8,
    ):
        super().__init__()
        self.use_scalar = use_scalar
        self.d_model = d_model
        self.n_predictions = n_predictions
        self.use_angle_embedding = use_angle_embedding

        # Input projection to model dimension
        self.in_proj = nn.Linear(input_features, d_model)
        self.ln_in = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        # Positional embedding (learned)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Scalar embedding (optional)
        if self.use_scalar:
            self.scalar_embedding = nn.Sequential(
                nn.Linear(1, scalar_embedding_dim),
                nn.ReLU(),
                nn.Linear(scalar_embedding_dim, scalar_embedding_dim),
            )
            self.scalar_proj = nn.Linear(d_model + scalar_embedding_dim, d_model)
        else:
            self.scalar_embedding = None
            self.scalar_proj = None

        # Transformer encoder (PyTorch)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.ln_enc = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        # Step-query attention (cross-attention from queries to encoded sequence)
        self.attention = StepQueryAttention(
            n_predictions=n_predictions,
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
        )

        if self.use_angle_embedding:
            if angle_embedding_dim <= 0:
                raise ValueError("angle_embedding_dim must be > 0 when use_angle_embedding=True")
            self.angle_embedding = nn.Embedding(n_predictions, angle_embedding_dim)
            fc_in_dim = d_model + angle_embedding_dim
        else:
            self.angle_embedding = None
            fc_in_dim = d_model

        # Prediction head
        mid1 = max(32, fc_in_dim // 2)
        mid2 = max(16, fc_in_dim // 4)
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, mid1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid1, mid2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid2, output_features),
        )

    def forward(
        self,
        x: torch.Tensor,
        scalar: torch.Tensor | None = None,
        config: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        """
        x: (B,T,F)
        mask: (B,T) bool, True=valid
        """
        B, T, _ = x.shape
        if T > self.pos_emb.num_embeddings:
            raise ValueError(
                f"Sequence length T={T} exceeds max_len={self.pos_emb.num_embeddings}. "
                "Increase max_len in the model."
            )

        # Project + add positional embeddings
        h = self.in_proj(x)  # (B,T,D)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # (B,T)
        h = h + self.pos_emb(pos)
        h = self.ln_in(h)

        # Optional scalar conditioning
        if self.use_scalar:
            if scalar is None:
                raise ValueError("scalar input required when use_scalar=True")
            if scalar.dim() == 1:
                scalar = scalar.unsqueeze(-1)  # (B,1)

            s = self.scalar_embedding(scalar)              # (B,S)
            s = s.unsqueeze(1).expand(-1, T, -1)          # (B,T,S)
            h = self.scalar_proj(torch.cat([h, s], dim=-1))  # (B,T,D)

        # Transformer expects src_key_padding_mask: True = PAD (ignore)
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask.bool()

        # Encode
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B,T,D)
        h = self.ln_enc(h)

        # Step-query attention over encoded sequence
        ctx, attn = self.attention(h, mask=mask)  # (B,A,D), (B,A,T)

        # Predict per horizon
        if self.use_angle_embedding:
            angle_idx = torch.arange(self.n_predictions, device=ctx.device)
            angle_emb = self.angle_embedding(angle_idx).unsqueeze(0).expand(ctx.size(0), -1, -1)
            ctx = torch.cat([ctx, angle_emb], dim=-1)
        out = self.fc(ctx)  # (B,A,output_features)
        return out, attn
