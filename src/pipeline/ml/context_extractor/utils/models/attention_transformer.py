from __future__ import annotations

import math
import torch
import torch.nn as nn


class StepQueryAngleAttention(nn.Module):
    """
    Single attention distribution per angle (prediction step) over input timesteps.

    Inputs:
      H:    (B, T, D)
      mask: (B, T) bool True=valid, False=pad (optional)

    Returns:
      ctx:  (B, A, D)
      attn: (B, A, T)
    """

    def __init__(self, n_angles: int, d_model: int, attn_dropout: float = 0.0):
        super().__init__()
        self.n_angles = n_angles
        self.d_model = d_model

        # One learnable query per angle
        self.angle_queries = nn.Parameter(torch.empty(n_angles, d_model))
        nn.init.xavier_uniform_(self.angle_queries)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, H: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, D = H.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got D={D}")

        # Q: (B, A, D)
        Q = self.angle_queries.unsqueeze(0).expand(B, -1, -1)

        # scores: (B, A, T) = Q @ H^T / sqrt(D)
        scores = torch.matmul(Q, H.transpose(1, 2)) / math.sqrt(D)

        if mask is not None:
            # mask True=valid, False=pad -> set pad positions to -inf
            # scores shape: (B,A,T); mask shape: (B,T) -> broadcast to (B,1,T)
            scores = scores.masked_fill(~mask.bool().unsqueeze(1), float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # (B, A, T)
        attn = self.dropout(attn)

        # ctx: (B, A, D) = attn @ H
        ctx = torch.matmul(attn, H)
        return ctx, attn


class TransformerAttention(nn.Module):
    """
    Transformer encoder + angle-query attention + optional scalar side input.

    Inputs:
      x:      (B, T, input_features)
      scalar: (B,) or (B,1) if use_scalar=True
      mask:   (B, T) bool True=valid, False=pad (optional)

    Returns:
      out:  (B, A, output_features)
      attn: (B, A, T)   <-- A is number of angles
    """

    def __init__(
        self,
        input_features: int,
        n_predictions: int,              # this is your number of angles A
        output_features: int = 1,
        d_model: int = 64,
        n_heads: int = 4,                # only for the TRANSFORMER encoder layers
        transformer_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,

        use_layernorm: bool = True,
        max_len: int = 4096,
        angle_attn_dropout: float = 0.0, # dropout on attention weights
    ):
        super().__init__()
        self.d_model = d_model
        self.n_predictions = n_predictions  # angles A

        self.in_proj = nn.Linear(input_features, d_model)
        self.ln_in = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.pos_emb = nn.Embedding(max_len, d_model)

    

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,  # encoder heads are fine; this is NOT your angle attention
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.ln_enc = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        # <<< This replaces MultiheadAttention completely >>>
        self.angle_attention = StepQueryAngleAttention(
            n_angles=n_predictions,
            d_model=d_model,
            attn_dropout=angle_attn_dropout,
        )

        mid1 = max(32, d_model // 2)
        mid2 = max(16, d_model // 4)
        self.fc = nn.Sequential(
            nn.Linear(d_model, mid1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid1, mid2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid2, output_features),
        )

    def forward(
        self,
        x: torch.Tensor,                   # (B,T,F)
        mask: torch.Tensor | None = None,  # (B,T) True=valid, False=pad
    ):
        B, T, _ = x.shape
        if T > self.pos_emb.num_embeddings:
            raise ValueError(
                f"Sequence length T={T} exceeds max_len={self.pos_emb.num_embeddings}. "
                "Increase max_len in the model."
            )

        h = self.in_proj(x)  # (B,T,D)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # (B,T)
        h = self.ln_in(h + self.pos_emb(pos))

        h = self.encoder(h)  # (B,T,D)
        h = self.ln_enc(h)

        # Angle attention: (B,A,D) and (B,A,T)
        ctx, attn = self.angle_attention(h, mask=mask)

        out = self.fc(ctx)  # (B,A,output_features)
        return out, attn
