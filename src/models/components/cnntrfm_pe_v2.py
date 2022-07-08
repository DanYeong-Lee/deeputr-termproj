from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


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

class CNNTRFM_v2(nn.Module):
    def __init__(
        self,
        kernel_sizes: List = [6, 9, 12, 15],
        out_channels: int = 256,
        pool_size: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048
    ):
        super().__init__()
        self.encoder = Encoder(kernel_sizes, out_channels, pool_size)
        self.fc1 = nn.Linear(len(kernel_sizes) * out_channels, d_model)
        self.pe = PositionalEncoding1D(d_model)
        self.decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.embed_tgt = nn.Linear(1, d_model)
        self.out = nn.Linear(d_model, 1)
        
        
    def forward(self, x, init_level):
        # x: (N, L, C)
        x = self.encoder(x)  # (N, C, L)
        x = x.transpose(1, 2)  # (N, L, C)
        x = self.fc1(x)  # (N, L, d_model)
        x = x + self.pe(x)  # Positional Encoding
        x = x.transpose(0, 1)  # (L, N, d_model)
        
        tgt = self.embed_tgt(init_level.unsqueeze(-1))  # (N, d_model)
        tgt = tgt.unsqueeze(0)  # (1, N, d_model)
        
        for _ in range(8):
            tgt = tgt.transpose(0, 1)  # (N, L, C)
            in_tgt = tgt + self.pe(tgt)  # (N, L, C)
            in_tgt = in_tgt.transpose(0, 1)
            out_tgt = self.decoder(in_tgt, x)  # (L, N, d_model)
            out = self.out(out_tgt[-1]) # (N, 1)
            next_tgt = self.embed_tgt(out).unsqueeze(0)  # (1, N, d_model)
            tgt = torch.cat([tgt.transpose(0, 1), next_tgt], axis=0)  # (L+1, N, d_model)
        
        outputs = torch.cat(outputs, axis=1)  # (N, 8)

        return outputs
    
    
