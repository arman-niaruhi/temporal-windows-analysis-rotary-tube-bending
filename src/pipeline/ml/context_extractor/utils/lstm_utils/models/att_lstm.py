import torch
import torch.nn as nn


class MLPAttention(nn.Module):
    def __init__(self, n_predictions, hidden_dim=128):
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
        input_features,
        n_predictions,
        output_features=1,
        hidden_dim=128,
        lstm_layers=2,
        dropout=0.3,
    ):
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
