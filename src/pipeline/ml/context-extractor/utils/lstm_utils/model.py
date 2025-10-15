import torch
import torch.nn as nn
import numpy as np

class AngleAwareLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=4, 
                 num_layers=1, num_angle_freqs=8, bidirectional=False):
        super().__init__()
        self.num_angle_freqs = num_angle_freqs
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM input: original features + angle embedding + ordinal
        self.lstm = nn.LSTM(
            input_size=input_dim + 2 * num_angle_freqs + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Learnable attention vector
        self.attn_vec = nn.Parameter(torch.randn(hidden_dim * self.num_directions, 1))

    def angle_embedding(self, angle):
        """
        High-frequency sinusoidal embedding for angles.
        angle: (batch, 1) normalized [0,1]
        returns: (batch, 2*num_angle_freqs)
        """
        embeddings = [torch.sin((2**i) * np.pi * angle) for i in range(self.num_angle_freqs)]
        embeddings += [torch.cos((2**i) * np.pi * angle) for i in range(self.num_angle_freqs)]
        return torch.cat(embeddings, dim=1)

    def forward(self, x, angle, angle_idx=None, num_angles=None, return_attention=False):
        """
        x: (batch, seq_len, num_features)
        angle: (batch,1) normalized [0,1]
        angle_idx: (batch,1) optional, integer index of the angle
        num_angles: total number of discrete angles (for normalization)
        """
        angle = angle.to(x.dtype)
        
        # Sinusoidal embedding
        angle_emb = self.angle_embedding(angle)
        
        # Optional ordinal feature
        if angle_idx is not None and num_angles is not None:
            ordinal = (angle_idx / (num_angles - 1)).to(x.dtype)  # normalized [0,1]
            angle_emb = torch.cat([angle_emb, ordinal], dim=1)
        else:
            # If not provided, just add a zero column to match LSTM input
            ordinal = torch.zeros(angle.size(0), 1, dtype=x.dtype, device=x.device)
            angle_emb = torch.cat([angle_emb, ordinal], dim=1)
        
        # Repeat across sequence
        angle_emb_seq = angle_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x_with_angle = torch.cat([x, angle_emb_seq], dim=2)
        
        # LSTM + attention
        lstm_out, _ = self.lstm(x_with_angle)
        attn_scores = torch.matmul(lstm_out, self.attn_vec).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        seq_repr = torch.sum(lstm_out * attn_weights, dim=1)
        
        out = self.fc(seq_repr)
        
        if return_attention:
            return out, attn_weights
        else:
            return out
