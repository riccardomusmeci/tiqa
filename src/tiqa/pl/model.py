from typing import Dict, List, Literal, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..utils import get_device


class IQAModelModule(pl.LightningModule):
    """IQA PyTorchLightning module that combines a model with a loss function,
    an optimizer, a learning rate scheduler, and evaluation metrics.

    Args:
        model (nn.Module): model to train.
        loss (_Loss): loss function to use during training.
        optimizer (Optimizer): optimizer to use during training.
        lr_scheduler (_LRScheduler): learning rate scheduler to use during training.
        unfreeze_after (int): number of epochs after which to unfreeze the model. Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: _Loss,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        unfreeze_after: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.unfreeze_after = unfreeze_after

        # SRCC and PLCC
        self.scores = []  # type: ignore
        self.targets = []  # type: ignore
        self.val_loss = []  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: logits
        """
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:  # type: ignore
        """Training step.

        Args:
            batch: batch
            batch_idx: batch idx


        Returns:
            torch.Tensor: train loss
        """

        x, target = batch

        logits = self(x)
        target = target.view(len(target), 1).to(self.device)
        loss = self.loss(logits, target)

        self.log("loss_train", loss, sync_dist=True, prog_bar=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        """Validation step.

        Args:
            batch: batch
            batch_idx: batch index
        """
        x, target = batch
        logits = self(x)
        target = target.view(len(target), 1).to(self.device)
        loss = self.loss(logits, target)

        # saving predictions and targets for SRCC and PLCC
        self.scores.extend(logits[:, 0].cpu().tolist())
        self.targets.extend(target.cpu().tolist())
        self.val_loss.append(loss)

    def on_validation_epoch_end(self):  # type: ignore
        """Validation epoch end."""
        # logging loss
        avg_loss = torch.tensor(self.val_loss).mean()
        self.log("loss_val", avg_loss, sync_dist=True, prog_bar=True)

        srcc = spearmanr(self.scores, self.targets)[0]
        plcc = pearsonr(self.scores, self.targets)[0][0]

        # resetting data structures
        self.scores = []
        self.targets = []
        self.val_loss = []

        srcc = torch.tensor(srcc, dtype=torch.float32)
        plcc = torch.tensor(plcc, dtype=torch.float32)

        # logging metrics
        self.log("srcc", srcc, sync_dist=True, prog_bar=True)
        self.log("plcc", plcc, sync_dist=True, prog_bar=True)

    def on_train_epoch_start(self):
        """Unfreeze model after a certain number of epochs."""
        if self.unfreeze_after is not None:
            if self.current_epoch == self.unfreeze_after:
                print(f"Unfreezing model at epoch {self.unfreeze_after}")
                self.model.unfreeze()

    def configure_optimizers(self) -> Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]:
        """Configure optimizer and lr scheduler.

        Returns:
            Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]: optimizer and lr scheduler
        """
        if self.lr_scheduler is None:
            return [self.optimizer]  # type: ignore
        else:
            return [self.optimizer], [self.lr_scheduler]
