from abc import ABC, abstractmethod
from typing import Any, Tuple

import matplotlib.pyplot as plt
import torch
import wandb
from einops import rearrange, repeat
from lightning import LightningModule
from torch import nn
from torch.distributions import Categorical
from torchmetrics import MeanMetric, SpearmanCorrCoef

from chunkgfn.gfn.utils import has_trainable_parameters
from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer
from chunkgfn.replay_buffer.utils import extend_trajectories
from chunkgfn.schedulers import Scheduler

from ..constants import EPS, NEGATIVE_INFINITY


class UnConditionalSequenceGFN(ABC, LightningModule):
    """Abstract class for sequence-based Generative Flow Networks."""

    def __init__(
        self,
        forward_model: torch.nn.Module,
        backward_model: torch.nn.Module,
        action_model: torch.nn.Module,
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
        self.action_model = action_model
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
        self.train_trajectory_length = MeanMetric()
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
        if self.action_model is not None and has_trainable_parameters(
            self.action_model
        ):
            params.append(
                {
                    "params": self.action_model.parameters(),
                    "lr": self.hparams.action_lr,
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

    @torch.no_grad()
    def refactor_replay_buffer(self):
        """Refactor the replay buffer. This function takes final states from the replay
        buffer and samples backward trajectories for them to get different trajctories
        based on the current library.
        """
        if self.replay_buffer is not None:
            final_state = self.replay_buffer.storage["final_state"]
            trajectories, actions, dones, _ = self.go_backward(final_state)
            self.replay_buffer.storage["trajectories"] = trajectories
            self.replay_buffer.storage["actions"] = actions
            self.replay_buffer.storage["dones"] = dones

    def get_library_embeddings(self):
        """Produce embedding for all actions in the library.
        Returns:
            library_embeddings (torch.Tensor[n_actions, action_embedding]): Embeddings for all actions.
        """
        action_indices = self.trainer.datamodule.action_indices
        library_embeddings = []
        for action, indices in action_indices.items():
            library_embeddings.append(
                self.action_model(
                    torch.LongTensor(indices).to(self.device).unsqueeze(0)
                )
            )
        library_embeddings = torch.cat(library_embeddings, dim=0)
        return library_embeddings

    def get_forward_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get the forward logits for the given state.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): State.
        Return:
            logits (torch.Tensor[batch_size, n_actions]): Forward logits.
        """
        action_embedding = self.forward_model(
            self.trainer.datamodule.preprocess_states(state)
        )
        dim = action_embedding.shape[-1]
        library_embeddings = self.get_library_embeddings()
        logits = torch.einsum("bd, nd -> bn", action_embedding, library_embeddings) / (
            dim**0.5
        )  # Same as in softmax
        return logits

    def go_backward(
        self,
        final_state: torch.Tensor,
    ):
        """Sample backward trajectories conditioned on inputs.
        Args:
            final_state (torch.Tensor[batch_size, *state_shape]): Final state.
        Return:
            trajectories (torch.Tensor[batch_size, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            state (torch.Tensor[batch_size, *state_shape]): Final state.
        """
        bs = final_state.shape[0]
        state = final_state.clone()
        done = torch.zeros((bs)).to(final_state).bool()

        # Start unrolling the trajectories
        actions = []
        trajectories = []
        dones = []
        dones.append(torch.ones((bs)).to(final_state).bool())

        while not done.all():
            backward_actions = self.trainer.datamodule.get_parent_actions(state)
            logp_b_s = torch.where(
                backward_actions == 1, torch.tensor(0.0), -torch.inf
            ).to(state)
            # When no action is available, just fill with uniform because
            # it won't be picked anyway in the backward_step.
            # Doing this avoids having nan when computing probabilities
            logp_b_s = torch.where(
                (logp_b_s == -torch.inf).all(dim=-1).unsqueeze(1),
                torch.tensor(0.0),
                logp_b_s,
            )
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
            trajectories (torch.Tensor[batch_size, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            state (torch.Tensor[batch_size, *state_shape]): Final state.
            trajectory_length (torch.Tensor[batch_size]): Length of the trajectory for each sample in the batch.
        """
        s0 = self.trainer.datamodule.s0.to(self.device)
        state = repeat(s0, " ... -> b ...", b=batch_size)
        bs = state.shape[0]

        # Start unrolling the trajectories
        actions = []
        trajectories = []
        dones = []
        done = torch.zeros((bs)).to(state).bool()
        trajectory_length = (
            torch.zeros((bs)).to(state).long()
        )  # This tracks the length of trajectory for each sample in the batch

        while not done.all():
            logit_pf = self.get_forward_logits(state)
            uniform_dist_probs = torch.ones_like(logit_pf).to(logit_pf)

            valid_actions_mask = self.trainer.datamodule.get_forward_mask(state)

            logit_pf = torch.where(
                valid_actions_mask,
                logit_pf,
                torch.tensor(NEGATIVE_INFINITY).to(logit_pf),
            )
            uniform_dist_probs = torch.where(
                valid_actions_mask,
                uniform_dist_probs,
                torch.tensor(0.0).to(uniform_dist_probs),
            )

            if train:
                if temperature is not None:
                    logits = logit_pf / (EPS + temperature)
                else:
                    logits = logit_pf
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
                cat = Categorical(logits=logit_pf)

            act = cat.sample()

            new_state, done = self.trainer.datamodule.forward_step(state, act)
            trajectory_length += ~done  # Increment the length of the trajectory for each sample in the batch as long it's not done.

            actions.append(act)
            trajectories.append(state)
            dones.append(done.clone())

            state = new_state.clone()

        trajectories.append(state)
        dones.append(torch.ones((bs)).to(state).bool())
        trajectories = torch.stack(trajectories, dim=1)
        actions = torch.stack(actions, dim=1)
        dones = torch.stack(dones, dim=1)

        return trajectories, actions, dones, state, trajectory_length

    @abstractmethod
    def compute_loss(self, trajectories, actions, dones, logreward):
        """Compute the loss for the model.
        Args:
            trajectories (torch.Tensor[batch_size, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            logreward (torch.Tensor[batch_size]): Log reward.
        """
        NotImplementedError

    @torch.no_grad()
    def get_ll(
        self, final_state: torch.Tensor, logreward: torch.Tensor
    ) -> torch.Tensor:
        """Get the log likelihood of the model for the given trajectories.
        Args:
            final_state (torch.Tensor[batch_size, *state_shape]): The examples for which we're computing the log-likelihood.
            logreward (torch.Tensor[batch_size]): Log reward.
        Return:
            log_pT (torch.Tensor[batch_size]): Log likelihood.
            trajectories (torch.Tensor[batch_size*n_trajectories, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size*n_trajectories, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size*n_trajectories, trajectory_length]): Whether the trajectory is done or not.
            logreward (torch.Tensor[batch_size*n_trajectories]): Log reward.
        """
        # Repeat the final_state n_trajectories times
        bs = final_state.shape[0]
        final_state = repeat(
            final_state, "b ... -> b n ...", n=self.hparams.n_trajectories
        )
        final_state = rearrange(final_state, "b n ... -> (b n) ...")
        logreward = repeat(logreward, "b -> b n", n=self.hparams.n_trajectories)
        logreward = rearrange(logreward, "b n ... -> (b n) ... ")
        trajectories, actions, dones, final_state = self.go_backward(final_state)

        # Calculate the log likelihood
        log_pf = 0
        log_pb = 0
        for t in range(trajectories.shape[1]):
            state = trajectories[:, t]
            logit_pf = self.get_forward_logits(state)
            if t < trajectories.shape[1] - 1:
                log_pf += (Categorical(logits=logit_pf).log_prob(actions[:, t])) * (
                    ~dones[:, t] + 0
                )
            if t > 0:
                backward_actions = self.trainer.datamodule.get_parent_actions(state)
                logp_b_s = torch.where(
                    backward_actions == 1, torch.tensor(0.0), -torch.inf
                ).to(logit_pf)
                # When no action is available, just fill with uniform because it won't be picked anyway in the backward_step. Doing this avoids having nan when computing probabilities
                logp_b_s = torch.where(
                    (logp_b_s == -torch.inf).all(dim=-1).unsqueeze(1),
                    torch.tensor(0.0),
                    logp_b_s,
                )
                log_pb += torch.where(
                    dones[:, t] | self.trainer.datamodule.is_initial_state(state),
                    torch.tensor(0.0),
                    Categorical(logits=logp_b_s).log_prob(actions[:, t - 1]),
                )
        log_pf = rearrange(log_pf, "(b n) ... -> b n ...", b=bs)
        log_pb = rearrange(log_pb, "(b n) ... -> b n ...", b=bs)
        log_pT = torch.logsumexp(log_pf - log_pb, dim=1) - torch.log(
            torch.tensor(self.hparams.n_trajectories)
        )

        return log_pT, trajectories, actions, dones, logreward

    def sample(
        self,
        batch: torch.Tensor,
        train: bool = True,
        epsilon: float = 0.0,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch

        trajectories, actions, dones, final_state, trajectory_length = self.forward(
            x.shape[0], train=train, epsilon=epsilon, temperature=temperature
        )

        logreward = self.trainer.datamodule.compute_logreward(final_state).to(
            final_state.device
        )
        return (
            x,
            trajectories,
            actions,
            dones,
            final_state,
            logreward,
            trajectory_length,
        )

    def log_action_histogram(self):
        """Log the action histogram."""
        normalizing_constant = self.trainer.datamodule.action_frequency.sum()
        action_frequency = [
            [freq / normalizing_constant, action]
            for (freq, action) in zip(
                self.trainer.datamodule.action_frequency,
                self.trainer.datamodule.actions,
            )
        ]
        table = wandb.Table(data=action_frequency, columns=["frequency", "action"])

        self.logger.log_metrics(
            {
                "action_histogram": wandb.plot.bar(
                    table, "action", "frequency", title="Action Frequency"
                )
            }
        )

    def training_step(self, train_batch, batch_idx) -> Any:
        if self.epsilon_scheduler is not None:
            epsilon = self.epsilon_scheduler.step(self.current_epoch)
        else:
            epsilon = None
        if self.temperature_scheduler is not None:
            temperature = self.temperature_scheduler.step(self.current_epoch)
        else:
            temperature = None

        x, trajectories, actions, dones, final_state, logreward, trajectory_length = (
            self.sample(
                train_batch,
                train=True,
                epsilon=epsilon,
                temperature=temperature,
            )
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
            trajectories, actions, dones = extend_trajectories(
                trajectories[indices],
                samples["trajectories"],
                actions[indices],
                samples["actions"],
                dones[indices],
                samples["dones"],
            )

            final_state = torch.cat(
                [final_state[indices], samples["final_state"]], dim=0
            )
            logreward = torch.cat([logreward[indices], samples["logreward"]], dim=0)

        loss, logZ = self.compute_loss(trajectories, actions, dones, logreward)
        additional_metrics = self.trainer.datamodule.compute_metrics(final_state)

        self.train_loss(loss)
        self.train_logreward(logreward.mean())
        self.train_logZ(logZ.mean())
        self.train_trajectory_length(trajectory_length.float().mean())

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
        if self.replay_buffer is not None:
            self.log(
                "replay_buffer_size",
                float(len(self.replay_buffer)),
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
        self.log(
            "train/trajectory_length",
            self.train_trajectory_length,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_action_histogram()

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
        log_pT, trajectories, actions, dones, logreward = self.get_ll(
            final_state, logreward
        )
        loss, _ = self.compute_loss(trajectories, actions, dones, logreward)

        logreward = rearrange(
            logreward, "(b n) ... -> b n ...", b=final_state.shape[0]
        ).mean(dim=1)

        self.val_loss(loss)
        self.val_logreward(logreward.mean())
        self.val_correlation(log_pT, logreward)

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
