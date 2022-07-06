from typing import Any, List

import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MinMetric, MeanSquaredError



class BaseNet(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 5e-4,
        weight_decay: float = 0,
        lr_schedule: bool = True,
        lr_stepsize: int = 3
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.criterion = nn.MSELoss()
        self.val_loss = MeanSquaredError()
        self.val_loss_best = MinMetric()
        
    def forward(self, x, init_level):
        return self.net(x, init_level)
    
    def on_train_start(self):
        self.val_loss_best.reset()
    
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
    
    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.val_loss.update(preds, targets)
        metrics = {"val/loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
    
    def validation_epoch_end(self, outputs):
        epoch_loss = self.val_loss.compute()
        self.val_loss_best.update(epoch_loss)
        self.log("val/loss_best", self.val_loss_best.compute(), on_epoch=True, prog_bar=True)
        self.val_loss.reset()
        
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
        if self.hparams.lr_schedule:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=range(9, 50, self.hparams.lr_stepsize),
                gamma=(1/np.exp(1)),
                verbose=True
            )

            return [optimizer], [scheduler]
        else:
            return optimizer

    
class MultiNet(BaseNet):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 5e-4,
        weight_decay: float = 0,
        lr_schedule: bool = True,
        lr_stepsize: int = 3
    ):
        super().__init__(net, lr, weight_decay, lr_schedule, lr_stepsize)
        
    def forward(self, x, plus_init, minus_init):
        return self.net(x, plus_init, minus_init)
    
    def step(self, batch):
        x, plus_init, minus_init, plus_y, minus_y = batch
        pred = self(x, plus_init, minus_init)
        y = torch.cat([plus_y, minus_y], axis=1)
        loss = self.criterion(pred, y)
        
        return loss, pred, y
    
    
class L2Net(BaseNet):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 5e-4,
        weight_decay: float = 0,
        lr_schedule: bool = True,
        l2_idx: List[int] = [0],
        lr_stepsize: int = 3
    ):
        super().__init__(net, lr, weight_decay, lr_schedule, lr_stepsize)
        self.l2_idx = l2_idx
        
        
    def training_step(self, batch, batch_idx):
        x, init_level, y = batch
        pred = self(x, init_level)
        loss = self.criterion(pred, y)
        params = list(self.parameters())
        for idx in self.l2_idx:
            l2_param = params[idx]
            l2_loss = torch.linalg.norm(l2_param)
            loss = loss + self.hparams.weight_decay * l2_loss
        
        metrics = {"train/loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=0
        )
        if self.hparams.lr_schedule:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=range(9, 50, self.hparams.lr_stepsize),
                gamma=(1/np.exp(1)),
                verbose=True
            )

            return [optimizer], [scheduler]
        else:
            return optimizer
    

class MultiL2Net(BaseNet):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 5e-4,
        weight_decay: float = 0,
        lr_schedule: bool = True,
        l2_idx: List[int] = [0],
        lr_stepsize: int = 3
    ):
        super().__init__(net, lr, weight_decay, lr_schedule, lr_stepsize)
        self.l2_idx = l2_idx
    
    def forward(self, x, plus_init, minus_init):
        return self.net(x, plus_init, minus_init)
    
    def step(self, batch):
        x, plus_init, minus_init, plus_y, minus_y = batch
        pred = self(x, plus_init, minus_init)
        y = torch.cat([plus_y, minus_y], axis=1)
        loss = self.criterion(pred, y)
        
        return loss, pred, y
        
    def training_step(self, batch, batch_idx):
        x, plus_init, minus_init, plus_y, minus_y = batch
        pred = self(x, plus_init, minus_init)
        y = torch.cat([plus_y, minus_y], axis=1)
        loss = self.criterion(pred, y)
        # L2 regularization
        params = list(self.parameters())
        for idx in self.l2_idx:
            l2_param = params[idx]
            l2_loss = torch.linalg.norm(l2_param)
            loss = loss + self.hparams.weight_decay * l2_loss
        
        metrics = {"train/loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    
class RMSpropNet(BaseNet):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 5e-4,
        weight_decay: float = 0,
        lr_schedule: bool = True,
        lr_stepsize: int = 3
    ):
        super().__init__(net, lr, weight_decay, lr_schedule, lr_stepsize)
    
    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=0
        )
        if self.hparams.lr_schedule:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=range(9, 50, self.hparams.lr_stepsize),
                gamma=(1/np.exp(1)),
                verbose=True
            )

            return [optimizer], [scheduler]
        else:
            return optimizer
        

class MultiNAdamNet(MultiNet):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 5e-4,
        weight_decay: float = 0,
        lr_schedule: bool = True,
        lr_stepsize: int = 3
    ):
        super().__init__(net, lr, weight_decay, lr_schedule, lr_stepsize)
        
    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=0
        )
        if self.hparams.lr_schedule:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=range(9, 50, self.hparams.lr_stepsize),
                gamma=(1/np.exp(1)),
                verbose=True
            )

            return [optimizer], [scheduler]
        else:
            return optimizer