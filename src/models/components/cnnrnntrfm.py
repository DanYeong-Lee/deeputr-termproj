from typing import List
import torch
from torch import nn
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

class Encoder(nn.Module):
    def __init__(
        self,
        kernel_sizes: List = [6, 9, 12, 15],
        out_channels: int = 256,
        pool_size: int = 3
    ):
        super().__init__() 
        self.conv_blocks = nn.ModuleList([ConvBlock(4, out_channels, k, pool_size) for k in kernel_sizes])
        
    def forward(self, x):
        # x: (N, L, C)
        x = x.transpose(1, 2)  # (N, C, L)
        conv_outs = []
        for conv in self.conv_blocks:
            conv_outs.append(conv(x))
        x = torch.cat(conv_outs, dim=1)  # (N, C, L)
        
        return x

class CNNRNNTRFM(nn.Module):
    def __init__(
        self,
        kernel_sizes: List = [6, 9, 12, 15],
        out_channels: int = 256,
        pool_size: int = 3,
        rnn_hidden_dim: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048
    ):
        super().__init__()
        self.encoder = Encoder(kernel_sizes, out_channels, pool_size)
        self.rnn = nn.GRU(input_size=len(kernel_sizes) * out_channels, hidden_size=rnn_hidden_dim, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_dim * 2, d_model)
        self.decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.embed_tgt = nn.Linear(1, d_model)
        self.out = nn.Linear(d_model, 1)
        
        
    def forward(self, x, init_level):
        # x: (N, L, C)
        x = self.encoder(x)  # (N, C, L)
        x = x.transpose(1, 2)  # (N, L, C)
        x, _ = self.rnn(x)  # (N, L, C)
        x = self.fc1(x)  # (N, L, d_model)
        x = x.transpose(0, 1)  # (L, N, d_model)
        tgt = self.embed_tgt(init_level.unsqueeze(-1))  # (N, d_model)
        tgt = tgt.unsqueeze(0)  # (1, N, d_model)
        
        outputs = []
        for _ in range(8):
            tgt = self.decoder(tgt, x)  # (L, N, d_model)
            out = self.out(tgt[-1]) # (N, 1)
            outputs.append(out)
            next_tgt = self.embed_tgt(out).unsqueeze(0)  # (1, N, d_model)
            tgt = torch.cat([tgt, next_tgt], axis=0)  # (L+1, N, d_model)
        
        outputs = torch.cat(outputs, axis=1)  # (N, 8)

        return outputs
    
    
class MultiCNNRNNTRFM(nn.Module):
    def __init__(
        self,
        kernel_sizes: List = [6, 9, 12, 15],
        out_channels: int = 256,
        pool_size: int = 3,
        rnn_hidden_dim: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048
    ):
        super().__init__()
        self.encoder = Encoder(kernel_sizes, out_channels, pool_size)
        self.rnn = nn.GRU(input_size=len(kernel_sizes) * out_channels, hidden_size=rnn_hidden_dim, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_dim * 2, d_model)
        self.decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.embed_tgt = nn.Linear(2, d_model)
        self.out = nn.Linear(d_model, 2)
        
        
    def forward(self, x, plus_init, minus_init):
        # x: (N, L, C)
        x = self.encoder(x)  # (N, C, L)
        x = x.transpose(1, 2)  # (N, L, C)
        x, _ = self.rnn(x)  # (N, L, C)
        x = self.fc1(x)  # (N, L, d_model)
        x = x.transpose(0, 1)  # (L, N, d_model)
        init_level = torch.cat([plus_init.unsqueeze(-1), minus_init.unsqueeze(-1)], axis=1)
        tgt = self.embed_tgt(init_level)  # (N, d_model)
        tgt = tgt.unsqueeze(0)  # (1, N, d_model)
                
        outputs = []
        for _ in range(8):
            tgt = self.decoder(tgt, x)  # (L, N, d_model)
            out = self.out(tgt[-1]) # (N, 2)
            outputs.append(out.unsqueeze(1)) # (N, 1, 2)
            next_tgt = self.embed_tgt(out).unsqueeze(0)  # (1, N, d_model)
            tgt = torch.cat([tgt, next_tgt], axis=0)  # (L+1, N, d_model)
        
        outputs = torch.cat(outputs, axis=1)  # (N, 8, 2)
        outputs = torch.cat([outputs[:, :, 0], outputs[:, :, 1]], axis=1)  # (N, 16)

        return outputs
        