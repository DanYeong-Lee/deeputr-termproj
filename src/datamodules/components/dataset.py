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
        init_level = row[2]
        target = row[3:].astype(float)
        X = self.seq2mat(seq)
        init_level = torch.tensor(init_level, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)
        
        return X, init_level, y