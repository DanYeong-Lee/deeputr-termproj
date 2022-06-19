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
        pool_size: int = 3
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )
    
    def forward(self, x):
        # x: (N, C, L)
        
        return self.main(x)


class DeepFamQ(nn.Module):
    def __init__(
        self,
        kernel_sizes: List = [6, 9, 12, 15],
        out_channels: int = 256,
        pool_size: int = 3,
        rnn_hidden_dim: int = 256,
        fc_dim: List[int] = [256, 1024, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        pool_out_len = int(1 + ((110 - pool_size) / pool_size))
        fc_input_dim = rnn_hidden_dim * 2 * pool_out_len
        
        self.conv_blocks = nn.ModuleList([ConvBlock(4, out_channels, k, pool_size) for k in kernel_sizes])
        self.rnn = nn.GRU(input_size=out_channels * len(kernel_sizes), hidden_size=rnn_hidden_dim, bidirectional=True)
        self.flat = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(fc_input_dim + 1, fc_dim[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fc_dim[i], fc_dim[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                for i in range(len(fc_dim) - 1)
            ]
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_dim[-1], 8)
        )
        
    def forward(self, x, init_level):
        # x: (N, L, C)
        x = x.transpose(1, 2)  # (N, C, L)
        conv_outs = []
        for conv in self.conv_blocks:
            conv_outs.append(conv(x))
        x = torch.cat(conv_outs, dim=1)  # (N, C, L)
        x = x.permute(2, 0, 1)  # (L, N, C)
        x, _ = self.rnn(x)  # (L, N, C)
        x = x.transpose(0, 1)  # (N, L, C)
        x = self.flat(x)  # (N, C)
        x = torch.cat([x, init_level.unsqueeze(1)], axis=1)
        x = self.fc1(x)
        for fc in self.fc_layers:
            x = fc(x)
        x = self.fc2(x)
        
        return x
    