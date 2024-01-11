from typing import Any, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torch.distributions import Categorical
from torchmetrics import MeanMetric

from chunkgfn.schedulers import Scheduler


class Cond_TBGFN(LightningModule):
    def __init__(
        self,
        forward_model: torch.nn.Module,
        partition_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        epsilon_scheduler: Scheduler | None = None,
        temperature_scheduler: Scheduler | None = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.forward_model = forward_model
        self.partition_model = partition_model
        self.criterion = criterion
        self.epsilon_scheduler = epsilon_scheduler
        self.temperature_scheduler = temperature_scheduler

        self.mse_loss = nn.MSELoss()

        self.train_loss = MeanMetric()
        self.train_logreward = MeanMetric()
        self.train_logZ = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_logreward = MeanMetric()
        self.val_logZ = MeanMetric()

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

    def forward(
        self,
        x: torch.Tensor,
        train: bool = True,
        epsilon: float | None = None,
        temperature: float | None = None,
    ):
        """Sample trajectories conditioned on inputs.
        Args:
            x (torch.Tensor[batch_size, seq_length, dim]): Conditioning vector.
            train (bool): Whether it's during train or eval. This makes sure that we don't sample off-policy during inference.
            epsilon (float|None): Epsilon value for epsilon greedy.
            temperature (float|None): Temperature value for tempering.
        Return:
            state (torch.Tensor[batch_size, seq_length, dim]): Generated state for each sample in the input.
            log_pf (torch.Tensor[batch_size]): Log forward probabilities for each trajectory.
            log_pb (torch.Tensor[batch_size]): Log backward probabilities for each trajectory.
            logZ (torch.Tensor[batch_size]): Log of the partition function for each sample in the input.
        """
        bs, max_len, dim = x.shape
        state = -torch.ones(bs, 1, dim).to(x)
        p_f_s = self.forward_model(x, state)
        log_pf = 0
        log_pb = 0

        # Start unrolling the trajectories
        for t in range(max_len):
            if train:
                if temperature is not None:
                    logits = p_f_s / (1e-6 + temperature)
                else:
                    logits = p_f_s
                if epsilon is not None:
                    probs = torch.softmax(logits, dim=-1)
                    uniform_dist_probs = torch.ones_like(probs).to(probs)
                    uniform_dist_probs = uniform_dist_probs / uniform_dist_probs.sum(
                        dim=-1, keepdim=True
                    )
                    probs = (1 - epsilon) * probs + epsilon * uniform_dist_probs
                    cat = Categorical(probs=probs)
                else:
                    cat = Categorical(logits=logits)
            else:
                cat = Categorical(logits=p_f_s)
            act = cat.sample()
            action = torch.zeros((bs, dim)).to(state)
            action.scatter_(-1, act.unsqueeze(-1), 1)
            action = action.unsqueeze(1)
            log_pf += Categorical(logits=p_f_s).log_prob(act)
            if t == 0:
                new_state = action
            else:
                new_state = torch.cat([state, action], dim=1)
            p_f_s = self.forward_model(x, new_state)
            state = new_state
        logZ = self.partition_model(x).squeeze(-1)
        return state, log_pf, log_pb, logZ

    def gfn_step(
        self,
        batch: torch.Tensor,
        train: bool = True,
        epsilon: float = 0.0,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch
        state, log_pf, log_pb, logZ = self(x, train, epsilon, temperature)
        logreward = self.criterion(state, x)
        loss = self.mse_loss(logZ + log_pf, logreward + log_pb)
        return loss, state, logreward, logZ, log_pf, log_pb

    @torch.no_grad
    def sample_trajectories(self, batch):
        logrewards = []
        log_pfs = []
        for _ in range(self.hparams.repetitions):
            _, _, logreward, _, log_pf = self.gfn_step(batch)
            logrewards.append(logreward)
            log_pfs.append(log_pf)
        logrewards = torch.stack(logrewards, dim=-1)
        log_pfs = torch.stack(log_pfs, dim=-1)
        return logrewards, log_pfs

    def training_step(self, train_batch, batch_idx) -> Any:
        if self.epsilon_scheduler is not None:
            epsilon = self.epsilon_scheduler.step(self.current_epoch)
        else:
            epsilon = None
        if self.temperature_scheduler is not None:
            temperature = self.temperature_scheduler.step(self.current_epoch)
        else:
            temperature = None

        loss, state, logreward, logZ, log_pf, log_pb = self.gfn_step(
            train_batch, train=True, epsilon=epsilon, temperature=temperature
        )

        self.train_loss(loss)
        self.train_logreward(logreward.mean())
        self.train_logZ(logZ.mean())

        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)
        self.log(
            "train/logreward",
            self.train_logreward,
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

    def validation_step(self, val_batch, batch_idx) -> Any:
        loss, state, logreward, logZ, log_pf, log_pb = self.gfn_step(
            val_batch, train=False, epsilon=None, temperature=None
        )

        self.val_loss(loss)
        self.val_logreward(logreward.mean())
        self.val_logZ(logZ.mean())

        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)
        self.log(
            "val/logreward",
            self.val_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/logZ",
            self.val_logZ,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
