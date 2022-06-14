import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from src.datamodules.components.dataset import BaseDataset



class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_dir: str = "/data/project/danyoung/deeputr-termproj/data/train.csv",
        val_dir: str = "/data/project/danyoung/deeputr-termproj/data/val.csv",
        test_dir: str = "/data/project/danyoung/deeputr-termproj/data/test.csv",
        batch_size: int = 96, 
        num_workers: int = 4
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        
        self.dataset = BaseDataset
    
    def setup(self, stage=None):
        if stage == "fit" or stage == None:
            import os
            print(os.getcwd())
            train_df = pd.read_csv(self.hparams.train_dir)
            val_df = pd.read_csv(self.hparams.val_dir)
            self.train_data = self.dataset(train_df)
            self.val_data = self.dataset(val_df)
        
        if stage == "test" or stage == None:
            test_df = pd.read_csv(self.hparams.test_dir)                
            self.test_data = self.dataset(test_df)
            
        if stage == "predict" or stage == None:
            predict_df = pd.read_csv(self.hparams.test_dir)
            self.predict_data = self.dataset(predict_df)
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )