import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
seq_in = 1743

# ---------------------------
# 2. Transformer Model
# ---------------------------
class TransformerEncoderWithAttention(nn.Module):
    def __init__(self, input_dim=17, output_dim=5, seq_out=47, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.seq_out = seq_out
        self.input_fc = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_in, d_model))

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True) for _ in range(num_layers)]
        )
        self.decoder_fc = nn.Linear(d_model, output_dim)
        self.attentions = []

    def forward(self, x):
        # x: (batch, seq_in, input_dim)
        self.attentions = []
        x = self.input_fc(x) + self.positional_encoding.unsqueeze(0)  # (batch, seq_in, d_model)
        for layer in self.layers:
            out = layer(x)
            # Save attention weights (approximate via attention matrix of first head)
            self.attentions.append(layer.self_attn.attn_output_weights.detach() if hasattr(layer.self_attn, 'attn_output_weights') else torch.zeros(1))
            x = out
        # Sample seq_out timestamps evenly
        indices = torch.linspace(0, x.size(1)-1, self.seq_out).long()
        out = x[:, indices, :]  # (batch, seq_out, d_model)
        out = self.decoder_fc(out)  # (batch, seq_out, output_dim)
        return out
