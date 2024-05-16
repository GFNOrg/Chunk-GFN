from abc import ABC, abstractmethod
from typing import Any, Tuple

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


class A2C(ABC, LightningModule):
    """Abstract class for RL algos."""

    def __init__(
        self,
        forward_model: torch.nn.Module,
        critic_model: torch.nn.Module,
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
        self.critic_model = critic_model
        self.action_model = action_model

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
        if self.critic_model is not None and has_trainable_parameters(
            self.critic_model
        ):
            params.append(
                {
                    "params": self.critic_model.parameters(),
                    "lr": self.hparams.critic_lr,
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

    def compute_loss(self, trajectories, actions, dones, logreward):
        """Compute the loss for the model.
        Args:
            trajectories (torch.Tensor[batch_size, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            logreward (torch.Tensor[batch_size]): Log reward.
        """

        # We don't compute the value and policy loss for the final state
        trajectories = trajectories[:, :-1]
        dones = dones[:, :-1]

        states = rearrange(trajectories, "b t ... -> (b t) ...")
        # Since reward only given at the end, return=reward
        returns = logreward.repeat_interleave(trajectories.shape[1]).exp()
        values = self.critic_model(
            self.trainer.datamodule.preprocess_states(states)
        ).squeeze()

        advantage = returns - values

        logits = self.get_forward_logits(states)
        forward_mask = self.trainer.datamodule.get_forward_mask(states)
        logits = torch.where(
            forward_mask,
            logits,
            torch.tensor(NEGATIVE_INFINITY).to(logits),
        )

        value_loss = advantage.pow(2)
        policy_loss = -advantage.detach() * Categorical(logits=logits).log_prob(
            rearrange(actions, "b t ... -> (b t) ...")
        )
        entropy_loss = Categorical(logits=logits).entropy()

        loss = rearrange(
            (value_loss + policy_loss - self.hparams.entropy_coeff * entropy_loss),
            "(b t) -> b t",
            b=trajectories.shape[0],
        )
        loss = torch.where(dones, 0, loss)
        loss = torch.clamp(loss.sum(1), max=5000, min=-5000)
        loss = loss.mean()

        return loss

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

        loss = self.compute_loss(trajectories, actions, dones, logreward)
        additional_metrics = self.trainer.datamodule.compute_metrics(final_state)

        self.train_loss(loss)
        self.train_logreward(logreward.mean())
        self.train_trajectory_length(trajectory_length.float().mean())

        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)
        self.log(
            "train/logreward",
            self.train_logreward,
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
        # self.log_action_histogram()

        for metric_name in additional_metrics:
            self.log(
                f"train/{metric_name}",
                additional_metrics[metric_name],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def on_validation_epoch_end(self):
        # Get on-policy samples from the policy
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
