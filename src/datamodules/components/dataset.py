import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(
        self, 
        df
    ):
        self.data = df.to_numpy()
        self.base2vec = {
            "A": [1., 0., 0., 0.],
            "T": [0., 1., 0., 0.],
            "C": [0., 0., 1., 0.],
            "G": [0., 0., 0., 1.]
        }
    
    def seq2mat(self, seq):
        mat = torch.tensor(list(map(lambda x: self.base2vec[x], seq)), dtype=torch.float32)
        return mat

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        seq = row[1]
        plus_init = row[2]
        plus_target = row[3:11].astype(float)
        
        X = self.seq2mat(seq)
        plus_init = torch.tensor(plus_init, dtype=torch.float32)
        plus_y = torch.tensor(plus_target, dtype=torch.float32)
        
        return X, plus_init, plus_y
    
    
class MultiDataset(Dataset):
    def __init__(
        self, 
        df
    ):
        self.data = df.to_numpy()
        self.base2vec = {
            "A": [1., 0., 0., 0.],
            "T": [0., 1., 0., 0.],
            "C": [0., 0., 1., 0.],
            "G": [0., 0., 0., 1.]
        }
    
    def seq2mat(self, seq):
        mat = torch.tensor(list(map(lambda x: self.base2vec[x], seq)), dtype=torch.float32)
        return mat

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        seq = row[1]
        plus_init = row[2]
        minus_init = row[11]
        plus_target = row[3:11].astype(float)
        minus_target = row[12:].astype(float)
        
        X = self.seq2mat(seq)
        plus_init = torch.tensor(plus_init, dtype=torch.float32)
        minus_init = torch.tensor(minus_init, dtype=torch.float32)
        plus_y = torch.tensor(plus_target, dtype=torch.float32)
        minus_y = torch.tensor(minus_target, dtype=torch.float32)
        
        return X, plus_init, minus_init, plus_y, minus_y