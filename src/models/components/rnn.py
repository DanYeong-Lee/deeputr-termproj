from typing import List
import torch
from torch import nn
import torch.nn.functional as F

    
class RNN(nn.Module):
    def __init__(
        self,
        hidden_size: int = 96,
        rnn_dropout: float = 0.5,
        fc_dim: List[int] = [512],
        fc_dropout: float = 0.1
    ):
        super().__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.gru_dropout = nn.Dropout(rnn_dropout)
        
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, fc_dim[0]),
            nn.ReLU(),
            nn.Dropout(fc_dropout)
        )
        
        self.fc_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fc_dim[i], fc_dim[i+1]),
                    nn.ReLU(),
                    nn.Dropout(fc_dropout)
                )
                for i in range(len(fc_dim) - 1)
            ]
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_dim[-1], 8)
        )
        
    def forward(self, x, init_level):
        # x: (N, L, C)
        _, x = self.gru(x)  # (2, N, C). Final hidden states from both direction
        x = x.transpose(0, 1)  # (N, 2, C)
        x = torch.flatten(x, start_dim=1)  # (N, C)
        x = self.gru_dropout(x)
        x = torch.cat([x, init_level.unsqueeze(1)], axis=1)
        x = self.fc1(x)
        for fc in self.fc_layers:
            x = fc(x)
        x = self.fc2(x)
        return x