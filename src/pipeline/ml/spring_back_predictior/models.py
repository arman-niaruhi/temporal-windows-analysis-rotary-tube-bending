import torch
import torch.nn as nn


class AttentionSpringbackLSTM(nn.Module):
    """
    LSTM with feature-wise temporal attention for springback prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
        fc_dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # ------------------
        # LSTM (UNIDIRECTIONAL)
        # ------------------
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ------------------
        # FEATURE-WISE ATTENTION
        # ------------------
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

        # ------------------
        # PREDICTION HEAD (RESIDUAL)
        # ------------------
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(fc_dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.norm2 = nn.LayerNorm(hidden_size // 2)
        self.drop2 = nn.Dropout(fc_dropout)

        self.out = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        """
        x: [batch, seq, features]
        """
        lstm_out, _ = self.lstm(x)  # [B, T, H]

        # Attention scores (feature-wise)
        attn_scores = self.attention(lstm_out)  # [B, T, H]
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Context vector
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [B, H]

        # Residual MLP
        h = self.fc1(context)
        h = self.act(h)
        h = self.norm1(h)
        h = self.drop1(h)

        h = h + context  # residual connection

        h = self.fc2(h)
        h = self.act(h)
        h = self.norm2(h)
        h = self.drop2(h)

        return self.out(h)
