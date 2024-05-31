from typing import Any, Tuple

import torch
import wandb
from einops import repeat
from lightning import LightningModule
from torch import nn
from torch.distributions import Categorical
from torchmetrics import MeanMetric, SpearmanCorrCoef

from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer
from chunkgfn.replay_buffer.utils import extend_trajectories

from ..constants import NEGATIVE_INFINITY


class RandomSampler(LightningModule):
    """Abstract class for sequence-based Generative Flow Networks."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer | None = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.replay_buffer = replay_buffer

        self.train_logreward = MeanMetric()
        self.val_logreward = MeanMetric()
        self.train_trajectory_length = MeanMetric()
        self.val_correlation = (
            SpearmanCorrCoef()
        )  # Correlation between likelihood and logreward

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

    def configure_optimizers(self):
        params = [
            {
                "params": nn.Parameter(torch.zeros(1)),
                "lr": 0,
            }
        ]  # dummy params for lightning not to complain

        optimizer = self.hparams.optimizer(params=params)
        return {"optimizer": optimizer}

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
            forward_actions = self.trainer.datamodule.get_forward_mask(state)
            logit_pf = torch.where(
                forward_actions == 1, torch.tensor(0.0), torch.tensor(NEGATIVE_INFINITY)
            ).to(state)
            uniform_dist_probs = torch.ones_like(logit_pf).to(logit_pf)

            uniform_dist_probs = torch.where(
                forward_actions,
                uniform_dist_probs,
                torch.tensor(0.0).to(uniform_dist_probs),
            )

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

    def sample(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch

        trajectories, actions, dones, final_state, trajectory_length = self.forward(
            x.shape[0]
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
        x, trajectories, actions, dones, final_state, logreward, trajectory_length = (
            self.sample(
                train_batch,
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

        additional_metrics = self.trainer.datamodule.compute_metrics(final_state)

        self.train_logreward(logreward.mean())
        self.train_trajectory_length(trajectory_length.float().mean())

        self.log(
            "train/logreward",
            self.train_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
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
        # self.log_action_histogram()

        for metric_name in additional_metrics:
            self.log(
                f"train/{metric_name}",
                additional_metrics[metric_name],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return None

    def on_validation_epoch_end(self):
        # Get on-policy samples from the GFN
        dummy_batch = torch.arange(self.hparams.n_onpolicy_samples).to(self.device)
        x, trajectories, actions, dones, final_state, logreward, trajectory_length = (
            self.sample(
                dummy_batch,
                train=False,
                epsilon=None,
                temperature=None,
            )
        )
        torch.save(
            {
                "actions": actions,
                "dones": dones,
                "final_state": final_state,
                "trajectory_length": trajectory_length,
                "epoch": self.current_epoch,
            },
            f"{self.trainer.log_dir}/on_policy_samples_{self.current_epoch}.pt",
        )

        # Save the library and frequency of use at each epoch
        action_frequency = [
            [freq, action]
            for (freq, action) in zip(
                self.trainer.datamodule.action_frequency,
                self.trainer.datamodule.actions,
            )
        ]
        torch.save(
            action_frequency, f"{self.trainer.log_dir}/library_{self.current_epoch}.pt"
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
