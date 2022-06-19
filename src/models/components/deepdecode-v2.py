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

    
class DeepDecode_v2(nn.Module):
    def __init__(
        self,
        kernel_sizes: List[int] = [6, 9, 12, 15],
        out_channels: int = 256,
        embed_dim: int = 256,
        fc_dim: List[int] = [1024, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv_blocks = nn.ModuleList([ConvBlock(i, out_channels) for i in kernel_sizes])
        self.embed = nn.Linear(len(kernel_sizes) * out_channels, embed_dim)
        self.gru_cell = nn.GRUCell(1, embed_dim)
        self.out = nn.Sequential(
            nn.Linear(embed_dim, fc_dim[0]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim[0], fc_dim[1]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim[1], 1)
        )
        
    def forward(self, x, init_level):
        # x: (N, L, C)
        x = x.transpose(1, 2)
        temp = []
        for conv in self.conv_blocks:
            temp.append(conv(x))
        x = torch.cat(temp, axis=1)
        h = self.embed(x)
        
        outputs = []
        out = init_level.unsqueeze(1)
        for _ in range(8):
            h = self.gru_cell(out, h)
            out = self.out(h)
            outputs.append(out)
        
        return torch.cat(outputs, axis=1)