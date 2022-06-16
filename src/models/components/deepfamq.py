from typing import List
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
            nn.Conv1d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(pool_size)
        )
    
    def forward(self, x):
        # x: (N, C, L)
        
        return self.main(x)


class DeepFamQ(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 256,
        conv_kernel_size: List = [6, 9],
        pool_size: int = 3,
        rnn_hidden_dim: int = 256,
        fc_hidden_dim: int = 64,
        dropout1: float = 0.2,
        dropout2: float = 0.5
    ):
        super().__init__()
        pool_out_len = int(1 + ((110 - pool_size) / pool_size))
        fc_input_dim = rnn_hidden_dim * 2
        conv_each_dim = int(conv_out_dim / len(conv_kernel_size))
        
        self.conv_blocks = nn.ModuleList([ConvBlock(4, conv_each_dim, k, pool_size, dropout1) for k in conv_kernel_size])
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
        conv_outs = []
        for conv in self.conv_blocks:
            conv_outs.append(conv(x))
        x = torch.cat(conv_outs, dim=1)  # (N, C, L)
        x = x.permute(2, 0, 1)  # (L, N, C)
        _, x = self.rnn(x)  # (2, N, C)
        x = x.transpose(0, 1)  # (N, 2, C)
        x = self.flat(x)  # (N, C)
        x = torch.cat([x, init_level.unsqueeze(1)], axis=1)
        x = self.fc(x)
        
        return x