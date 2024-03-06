from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from einops import rearrange, repeat
from lightning import LightningModule
from torch import nn
from torch.distributions import Categorical
from torchmetrics import MeanMetric, SpearmanCorrCoef

from chunkgfn.gfn.utils import has_trainable_parameters
from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer
from chunkgfn.schedulers import Scheduler

NEG_INF = -1e6  # Negative infinity
SMALL_VALUE = 1e-6  # Small value to avoid division by zero


class UnConditionalSequenceGFN(ABC, LightningModule):
    """Abstract class for sequence-based Generative Flow Networks."""

    def __init__(
        self,
        forward_model: torch.nn.Module,
        backward_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        epsilon_scheduler: Scheduler | None = None,
        temperature_scheduler: Scheduler | None = None,
        replay_buffer: ReplayBuffer | None = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # GFlowNet modules
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.logZ = nn.Parameter(torch.zeros(1))

        # Schedulers for off-policy training
        self.epsilon_scheduler = epsilon_scheduler
        self.temperature_scheduler = temperature_scheduler

        self.replay_buffer = replay_buffer

        # Metric managers
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_logreward = MeanMetric()
        self.val_logreward = MeanMetric()
        self.train_logZ = MeanMetric()
        self.val_correlation = (
            SpearmanCorrCoef()
        )  # Correlation between likelihood and logreward

    def configure_optimizers(self):
        params = []
        if self.forward_model is not None and has_trainable_parameters(
            self.forward_model
        ):
            params.append(
                {
                    "params": self.forward_model.parameters(),
                    "lr": self.hparams.forward_lr,
                }
            )
        if self.backward_model is not None and has_trainable_parameters(
            self.backward_model
        ):
            params.append(
                {
                    "params": self.backward_model.parameters(),
                    "lr": self.hparams.backward_lr,
                }
            )

        params.append(
            {
                "params": self.logZ,
                "lr": self.hparams.partition_lr,
            }
        )

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

    def go_backward(
        self,
        final_state: torch.Tensor,
    ):
        """Sample backward trajectories conditioned on inputs.
        Args:
            final_state (torch.Tensor[batch_size, seq_length, input_dim]): Final state.
        Return:
            trajectories (torch.Tensor[batch_size, trajectory_length, seq_length, state_dim]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            state (torch.Tensor[batch_size, seq_length, state_dim]): Final state.
        """
        bs, max_len, dim = final_state.shape
        state = final_state.clone()
        done = torch.ones((bs)).to(final_state).bool()

        # Start unrolling the trajectories
        actions = []
        trajectories = []
        dones = []
        dones.append(torch.ones((bs)).to(final_state).bool())

        for t in range(max_len):
            backward_actions = self.trainer.datamodule.get_parent_actions(state)
            logp_b_s = torch.where(
                backward_actions == 1, torch.tensor(0.0), -torch.inf
            ).to(state)
            cat = Categorical(logits=logp_b_s)

            act = cat.sample()

            new_state, done = self.trainer.datamodule.backward_step(state, act)

            actions.append(act)
            trajectories.append(state)
            dones.append(done.clone())

            state = new_state.clone()

        trajectories.append(state)

        trajectories = torch.stack(trajectories[::-1], dim=1)
        actions = torch.stack(actions[::-1], dim=1)
        dones = torch.stack(dones[::-1], dim=1)

        return trajectories, actions, dones, final_state

    def forward(
        self,
        batch_size: int,
        train: bool = True,
        epsilon: float | None = None,
        temperature: float | None = None,
    ):
        """Sample forward trajectories conditioned on inputs.
        Args:
            batch_size (int): Number of samples to generate.
            train (bool): Whether it's during train or eval. This makes sure that we don't sample off-policy during inference.
            epsilon (float|None): Epsilon value for epsilon greedy.
            temperature (float|None): Temperature value for tempering.
        Return:
            trajectories (torch.Tensor[batch_size, trajectory_length, seq_length, state_dim]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            state (torch.Tensor[batch_size, seq_length, state_dim]): Final state.
        """
        state = self.trainer.datamodule.s0.repeat(batch_size, 1, 1).to(self.device)
        bs, max_len, dim = state.shape

        # Start unrolling the trajectories
        actions = []
        trajectories = []
        dones = []

        for t in range(max_len):
            p_f_s = self.forward_model(state)
            uniform_dist_probs = torch.ones_like(p_f_s).to(p_f_s)

            valid_actions_mask = self.trainer.datamodule.get_invalid_actions_mask(state)

            p_f_s = torch.where(valid_actions_mask, p_f_s, torch.tensor(NEG_INF))
            uniform_dist_probs = torch.where(
                valid_actions_mask, uniform_dist_probs, torch.tensor(0)
            )

            if train:
                if temperature is not None:
                    logits = p_f_s / (SMALL_VALUE + temperature)
                else:
                    logits = p_f_s
                if epsilon is not None:
                    probs = torch.softmax(logits, dim=-1)
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

            new_state, done = self.trainer.datamodule.forward_step(state, act)

            actions.append(act)
            trajectories.append(state)
            dones.append(done.clone())

            state = new_state.clone()

        trajectories.append(state)
        dones.append(torch.ones((bs)).to(state).bool())
        trajectories = torch.stack(trajectories, dim=1)
        actions = torch.stack(actions, dim=1)
        dones = torch.stack(dones, dim=1)

        return trajectories, actions, dones, state

    @abstractmethod
    def compute_loss(self, trajectories, actions, dones, logreward):
        """Compute the loss for the model.
        Args:
            trajectories (torch.Tensor[batch_size, trajectory_length, seq_length, state_dim]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length, state_dim]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            logreward (torch.Tensor[batch_size]): Log reward.
        """
        NotImplementedError

    @torch.no_grad()
    def get_ll(
        self, trajectories: torch.Tensor, actions: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Get the log likelihood of the model for the given trajectories.
        Args:
            trajectories (torch.Tensor[batch_size, trajectory_length, seq_length, state_dim]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
        Return:
            log_pf (torch.Tensor[batch_size]): Log likelihood.
        """
        log_pf = 0
        for t in range(trajectories.shape[1]):
            state = trajectories[:, t]
            logp_f_s = self.forward_model(state)
            if t < trajectories.shape[1] - 1:
                log_pf += (Categorical(logits=logp_f_s).log_prob(actions[:, t])) * (
                    ~dones[:, t] + 0
                )
        return log_pf

    def sample(
        self,
        batch: torch.Tensor,
        train: bool = True,
        epsilon: float = 0.0,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch

        trajectories, actions, dones, final_state = self.forward(
            x.shape[0], train=train, epsilon=epsilon, temperature=temperature
        )

        logreward = self.trainer.datamodule.compute_logreward(final_state).to(
            final_state
        )
        return x, trajectories, actions, dones, final_state, logreward

    def training_step(self, train_batch, batch_idx) -> Any:
        if self.epsilon_scheduler is not None:
            epsilon = self.epsilon_scheduler.step(self.current_epoch)
        else:
            epsilon = None
        if self.temperature_scheduler is not None:
            temperature = self.temperature_scheduler.step(self.current_epoch)
        else:
            temperature = None

        x, trajectories, actions, dones, final_state, logreward = self.sample(
            train_batch,
            train=True,
            epsilon=epsilon,
            temperature=temperature,
        )
        batch_size = x.shape[0]
        nsamples_replay = int(batch_size * self.hparams.ratio_from_replay_buffer)

        if self.replay_buffer is not None:
            with torch.no_grad():
                self.replay_buffer.add(
                    input=x,
                    trajectories=trajectories,
                    actions=actions,
                    dones=dones,
                    final_state=final_state,
                    logreward=logreward,
                )
                samples = self.replay_buffer.sample(nsamples_replay)

            for key in samples.keys():
                samples[key] = samples[key].to(x.device)

            # Concatenate samples from the replay buffer and the on-policy samples
            indices = torch.randperm(len(x))[: batch_size - nsamples_replay]
            x = torch.cat([x[indices], samples["input"]], dim=0)
            trajectories = torch.cat(
                [trajectories[indices], samples["trajectories"]], dim=0
            )
            actions = torch.cat([actions[indices], samples["actions"]], dim=0)
            dones = torch.cat([dones[indices], samples["dones"]], dim=0)
            final_state = torch.cat(
                [final_state[indices], samples["final_state"]], dim=0
            )
            logreward = torch.cat([logreward[indices], samples["logreward"]], dim=0)

        loss, logZ = self.compute_loss(trajectories, actions, dones, logreward)
        additional_metrics = self.trainer.datamodule.compute_metrics(final_state)

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

        if self.epsilon_scheduler is not None:
            self.log("epsilon", epsilon, on_step=False, on_epoch=True, prog_bar=False)
        if self.temperature_scheduler is not None:
            self.log(
                "temperature", epsilon, on_step=False, on_epoch=True, prog_bar=False
            )

        self.log(
            "replay_buffer_size",
            len(self.replay_buffer),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "replay_buffer_mean_logreward",
            self.replay_buffer.storage["logreward"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for metric_name in additional_metrics:
            self.log(
                f"train/{metric_name}",
                additional_metrics[metric_name],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, val_batch, batch_idx) -> Any:
        final_state, logreward = val_batch
        # Repeat the final_state n_trajectories times
        final_state = repeat(
            final_state, "b ... -> b n ...", n=self.hparams.n_trajectories
        )
        final_state = rearrange(final_state, "b n ... -> (b n) ...")

        trajectories, actions, dones, final_state = self.go_backward(final_state)
        likelihood = self.get_ll(trajectories, actions, dones)
        # Computes the likelihood by computing expectation over PB trajectories
        likelihood = rearrange(
            likelihood, "(b n) ... -> b n ...", b=final_state.shape[0]
        ).mean(dim=1)
        loss, _ = self.compute_loss(trajectories, actions, dones, logreward)

        self.val_loss(loss)
        self.val_logreward(logreward.mean())
        self.val_correlation(likelihood, logreward)

        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)
        self.log(
            "val/logreward",
            self.val_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/correlation",
            self.val_correlation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_save_checkpoint(self, checkpoint):
        """Add the replay buffer to the checkpoint.
        Args:
            checkpoint (dict): Checkpoint dictionary.
        """
        checkpoint["replay_buffer"] = self.replay_buffer

    def on_load_checkpoint(self, checkpoint):
        """Load the replay buffer from the checkpoint.
        Args:
            checkpoint (dict): Checkpoint dictionary.
        """
        self.replay_buffer = checkpoint["replay_buffer"]
