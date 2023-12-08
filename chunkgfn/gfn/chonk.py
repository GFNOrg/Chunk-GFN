from typing import Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from lightning import LightningModule
from torch import Tensor, nn
from torch.distributions import Categorical
from torchmetrics import MeanMetric, R2Score


class ChonkGFN(LightningModule):
    def __init__(
        self,
        forward_model: torch.nn.Module,
        partition_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.forward_model = forward_model
        self.partition_model = partition_model
        self.criterion = criterion

        self.mse_loss = nn.MSELoss()
        self.train_loss = MeanMetric()
        self.train_logreward = MeanMetric()
        self.train_logZ = MeanMetric()
        self.train_correlation = R2Score()

    def configure_optimizers(self):
        params = [
            {
                "params": self.forward_model.parameters(),
                "lr": self.hparams.lr,
            },
            {
                "params": self.partition_model.parameters(),
                "lr": self.hparams.partition_lr,
            },
        ]
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def forward(self, x: torch.Tensor):
        """Sample trajectories conditioned on inputs.
        Args:
            x (torch.Tensor[batch_size, seq_length, dim]): Conditioning vector.
        Return:
            state (torch.Tensor[batch_size, seq_length, dim]): Generated state for each sample in the input.
            log_pf (torch.Tensor[batch_size]): Log forward probabilities for each trajectory.
            logZ (torch.Tensor[batch_size]): Log of the partition function for each sample in the input.
        """
        bs, max_len, dim = x.shape
        state = -torch.ones(bs, 1, dim).to(x)
        p_f_s = self.forward_model(x, state)
        log_pf = 0

        # Start unrolling the trajectories
        for t in range(max_len):
            cat = Categorical(logits=p_f_s)
            act = cat.sample()
            action = torch.zeros((bs, dim)).to(state)
            action.scatter_(-1, act.unsqueeze(-1), 1)
            action = action.unsqueeze(1)
            log_pf += cat.log_prob(act)
            if t == 0:
                new_state = action
            else:
                new_state = torch.cat([state, action], dim=1)
            p_f_s = self.forward_model(x, new_state)
            state = new_state
        logZ = self.partition_model(x).squeeze(-1)
        return state, log_pf, logZ

    def gfn_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch
        state, log_pf, logZ = self(x)
        logreward = self.criterion(state, x)
        loss = self.mse_loss(logZ + log_pf, logreward)
        return loss, state, logreward, logZ, log_pf

    def training_step(self, train_batch, batch_idx) -> Any:
        loss, state, logreward, logZ, log_pf = self.gfn_step(train_batch)

        self.train_loss(loss)
        self.train_logreward(logreward.mean())
        self.train_logZ(logZ.mean())
        self.train_correlation(log_pf, logreward)

        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)
        self.log(
            "train/logreward",
            self.train_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/correlation",
            self.train_correlation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/logZ",
            self.train_logZ,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss
