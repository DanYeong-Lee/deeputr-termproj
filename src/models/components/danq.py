import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        out_dim: int = 256,
        kernel_size: int = 9,
        pool_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(pool_size)
        )
    
    def forward(self, x):
        # x: (N, C, L)
        
        return self.main(x)


class DanQ(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 256,
        conv_kernel_size: int = 9,
        pool_size: int = 3,
        rnn_hidden_dim: int = 256,
        fc_hidden_dim: int = 64,
        dropout1: float = 0.2,
        dropout2: float = 0.5
    ):
        super().__init__()
        conv_out_len = 110 - conv_kernel_size + 1
        pool_out_len = int(1 + ((conv_out_len - pool_size) / pool_size))
        fc_input_dim = rnn_hidden_dim * 2
        
        self.conv_block = ConvBlock(4, conv_out_dim, conv_kernel_size, pool_size, dropout1)

        self.rnn = nn.GRU(input_size=conv_out_dim, hidden_size=rnn_hidden_dim, bidirectional=True)
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim + 1, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(fc_hidden_dim, 8)
        )
        
    def forward(self, x, init_level):
        # x: (N, L, C)
        
        x = x.transpose(1, 2)  # (N, C, L)
        x = self.conv_block(x)
        x = x.permute(2, 0, 1)  # (L, N, C)
        _, x = self.rnn(x)  # (2, N, C)
        x = x.transpose(0, 1)  # (N, 2, C)
        x = self.flat(x)  # (N, C)
        x = torch.cat([x, init_level.unsqueeze(1)], axis=1)
        x = self.fc(x)
        
        return x
    