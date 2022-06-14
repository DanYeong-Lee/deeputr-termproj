from typing import List
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        kernel_size: int = 9,
        out_channels: int = 256
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(110 - kernel_size + 1)
        )
    
    def forward(self, x):
        # x: (N, C, L)
        return self.conv(x).squeeze(-1)

    
class CNN(nn.Module):
    def __init__(
        self,
        kernel_size: int = 9,
        out_channels: int = 256,
        fc_dim: List[int] = [256, 1024, 64]
    ):
        super().__init__()
        self.conv_block = ConvBlock(kernel_size, out_channels)
        self.fc1 = nn.Sequential(
            nn.Linear(out_channels + 1, fc_dim[0]),
            nn.ReLU()
        )
        
        self.fc_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fc_dim[i], fc_dim[i+1]),
                    nn.ReLU()    
                )
                for i in range(len(fc_dim) - 1)
            ]
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_dim[-1], 8)
        )
        
    def forward(self, x, init_level):
        # x: (N, L, C)
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = torch.cat([x, init_level.unsqueeze(1)], axis=1)
        x = self.fc1(x)
        for fc in self.fc_layers:
            x = fc(x)
        x = self.fc2(x)
        return x