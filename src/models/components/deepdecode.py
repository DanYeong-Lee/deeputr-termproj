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

    
class DeepDecode(nn.Module):
    def __init__(
        self,
        kernel_sizes: List[int] = [6, 9, 12, 15],
        out_channels: int = 256,
        embed_dim: int = 256
    ):
        super().__init__()
        self.conv_blocks = nn.ModuleList([ConvBlock(i, out_channels) for i in kernel_sizes])
        self.embed = nn.Linear(len(kernel_sizes) * out_channels + 1, embed_dim)
        self.gru_cell = nn.GRUCell(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, 1)
        
    def forward(self, x, init_level):
        # x: (N, L, C)
        x = x.transpose(1, 2)
        temp = []
        for conv in self.conv_blocks:
            temp.append(conv(x))
        temp.append(init_level.unsqueeze(-1))
        x = torch.cat(temp, axis=1)
        x = self.embed(x)
        outputs = []
        for _ in range(8):
            x = self.gru_cell(x)
            output = self.out(x)
            outputs.append(output)
            
        return torch.cat(outputs, axis=1)