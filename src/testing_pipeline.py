import os
import json
from typing import List
import numpy as np
import hydra
from omegaconf import DictConfig
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase
from torchmetrics.functional import pearson_corrcoef
from torch.nn.functional import mse_loss

from src import utils

log = utils.get_logger(__name__)


def evaluate(preds, targets):
    per_reporter_rmse = []
    per_reporter_corr = []
    per_timepoint_rmse = []
    per_timepoint_corr = []
    for i in range(preds.size(0)):
        rmse = mse_loss(preds[i], targets[i]).sqrt().item()
        corr = pearson_corrcoef(preds[i], targets[i]).item()
        per_reporter_rmse.append(rmse)
        per_reporter_corr.append(corr)
    for j in range(preds.size(1)):
        rmse = mse_loss(preds[:, j], targets[:, j]).sqrt().item()
        corr = pearson_corrcoef(preds[:, j], targets[:, j]).item()
        per_timepoint_rmse.append(rmse)
        per_timepoint_corr.append(corr)
    per_reporter_rmse = np.array(per_reporter_rmse)
    per_reporter_corr = np.array(per_reporter_corr)
    per_timepoint_rmse = np.array(per_timepoint_rmse)
    per_timepoint_corr = np.array(per_timepoint_corr)
    
    return per_reporter_rmse, per_reporter_corr, per_timepoint_rmse, per_timepoint_corr

def test(config: DictConfig) -> None:
    """Contains minimal example of the testing pipeline. Evaluates given checkpoint on a testset.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # Log hyperparameters
    if trainer.logger:
        trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path})

    log.info("Starting testing!")
    name = config.get("name")
    seed = config.get("seed")
    
    if type(seed) == int:
        seed = [seed]
        
    preds = []
    root_dir = config.get("original_work_dir")
    for s in seed:
        ckpt_path = f"{root_dir}/logs/experiments/runs/{name}/ckpts/seed{s}.ckpt"
        results = trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        temp_pred, temp_target = [], []
        for pred, target in results:
            temp_pred.append(pred)
            temp_target.append(target)
            
        preds.append(torch.cat(temp_pred, axis=0).unsqueeze(-1))
        targets = torch.cat(temp_target, axis=0)
    preds = torch.cat(preds, axis=-1).mean(-1)
    
    per_reporter_rmse, per_reporter_corr, per_timepoint_rmse, per_timepoint_corr = evaluate(preds, targets)
    print(per_reporter_rmse.mean())
    print(per_timepoint_corr.mean())
    