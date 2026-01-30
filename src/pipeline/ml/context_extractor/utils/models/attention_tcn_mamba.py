import torch
import torch.nn as nn

from src.pipeline.ml.context_extractor.utils.models.attention_lstm import MLPAttention
from src.pipeline.ml.context_extractor.utils.models.attention_tcn import TemporalBlock
from src.pipeline.ml.context_extractor.utils.models.attention_mamba import MambaEncoderCPU


class AttentionTCNMamba(nn.Module):
    def __init__(
        self,
        input_features: int,
        n_predictions: int,
        output_features: int = 1,
        hidden_dim: int = 128,
        tcn_layers: int = 4,
        mamba_layers: int = 2,
        d_state: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.1,
        scalar_embedding_dim: int = 16,
        use_scalar: bool = True,
        config_dim: int | None = None,
        config_embedding_dim: int = 16,
        use_config: bool = True,
        split_output_heads: bool = False,
        main_head_hidden_sizes: list[int] | None = None,
        secondary_head_hidden_sizes: list[int] | None = None,
        use_angle_embedding: bool = False,
        angle_embedding_dim: int = 8,
    ):
        super().__init__()
        self.use_scalar = use_scalar
        self.use_config = use_config
        self.n_predictions = n_predictions
        self.hidden_dim = hidden_dim
        self.output_features = output_features
        self.split_output_heads = split_output_heads
        self.use_angle_embedding = use_angle_embedding

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

        blocks: list[nn.Module] = []
        in_channels = input_features
        for i in range(max(1, tcn_layers)):
            dilation = 2**i
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = hidden_dim
        self.tcn = nn.Sequential(*blocks)

        self.mamba = MambaEncoderCPU(
            d_model=hidden_dim,
            n_layers=mamba_layers,
            d_state=d_state,
            dropout=dropout,
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
        if x.is_cuda:
            raise RuntimeError("This AttentionTCNMambaCPU is CPU-only. Move inputs/model to CPU.")

        attention_mask = mask
        if attention_mask is None:
            attention_mask = x.abs().sum(dim=-1) > 0

        h = x.transpose(1, 2)
        h = self.tcn(h)
        h = h.transpose(1, 2)

        h = self.mamba(h)
        h = self.ln(h)

        ctx, attn = self.attention(h, mask=attention_mask)

        combined = ctx
        if self.use_scalar:
            if scalar is None:
                raise ValueError("scalar input is required when use_scalar=True")
            if scalar.dim() == 1:
                scalar = scalar.unsqueeze(-1)
            scalar_emb = self.scalar_embedding(scalar)
            scalar_emb = scalar_emb.unsqueeze(1).expand(-1, self.n_predictions, -1)
            combined = torch.cat([combined, scalar_emb], dim=-1)

        if self.use_config:
            if config is None:
                raise ValueError("config input is required when use_config=True")
            if config.dim() == 1:
                config = config.unsqueeze(-1)
            config_emb = self.config_embedding(config)
            config_emb = config_emb.unsqueeze(1).expand(-1, self.n_predictions, -1)
            combined = torch.cat([combined, config_emb], dim=-1)

        if self.use_angle_embedding:
            angle_idx = torch.arange(self.n_predictions, device=combined.device)
            angle_emb = self.angle_embedding(angle_idx).unsqueeze(0).expand(combined.size(0), -1, -1)
            combined = torch.cat([combined, angle_emb], dim=-1)

        if self.fc_heads is not None:
            outs = [head(combined) for head in self.fc_heads]
            out = torch.cat(outs, dim=-1)
        else:
            out = self.fc(combined)

        return out, attn
