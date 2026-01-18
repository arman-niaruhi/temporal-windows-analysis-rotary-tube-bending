import torch
import torch.nn as nn

class AttentionSprigbackLSTM(nn.Module):
    """
    LSTM with attention mechanism for springback prediction.
    Attention helps the model focus on important time steps.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
        fc_dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(fc_dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.LayerNorm(hidden_size // 2)
        self.dropout2 = nn.Dropout(fc_dropout)
        
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        """
        Forward pass with attention mechanism.
        
        Args:
            x: Input tensor of shape [batch, timesteps, features]
        
        Returns:
            Output tensor of shape [batch, output_size]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden*directions]
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # [batch, seq, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)  # [batch, seq, 1]
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, hidden*directions]
        
        # Fully connected layers
        out = self.fc1(context)
        out = self.act(out) 
        out = self.bn1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.act(out) 
        out = self.bn2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        return out