from typing import Any, List

import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl


class BaseNet(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 5e-4,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.criterion = nn.MSELoss()
        
    def forward(self, x, init_level):
        return self.net(x, init_level)
    
    def step(self, batch):
        x, init_level, y = batch
        pred = self(x, init_level)
        loss = self.criterion(pred, y)
        
        return loss, pred, y
    
    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        metrics = {"train/loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch.step()
    
    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        metrics = {"val/loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        metrics = {"test/loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
    def predict_step(self, batch, batch_idx):
        _, preds, targets = self.step(batch)
        
        return preds, targets
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=(1/np.exp(1)),
            verbose=True
        )
        
        return [optimizer], [scheduler]
    
    
    
class L2Net(BaseNet):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 5e-4,
        weight_decay: float = 0,
        l2_idx: int = 0
    ):
        super().__init__(net, lr, weight_decay)
        self.l2_idx = l2_idx
        
        
    def step(self, batch):
        x, init_level, y = batch
        pred = self(x, init_level)
        loss = self.criterion(pred, y)
        l2_param = list(self.parameters())[self.l2_idx]
        l2_loss = torch.linalg.norm(l2_param)
        loss = loss + self.weight_decay * l2_loss
        
        return loss, pred, y