from copy import deepcopy
from typing import Any

import torch
from einops import rearrange
from torch import nn
from torch.distributions import Categorical

from chunkgfn.algo.sampler_base import BaseSampler
from chunkgfn.algo.utils import has_trainable_parameters
from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer
from chunkgfn.schedulers import Scheduler

from ..constants import EPS, NEGATIVE_INFINITY


class SAC(BaseSampler):
    """Soft Actor Critic module."""

    def __init__(
        self,
        forward_policy: nn.Module,
        action_embedder: nn.Module,
        critic: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        epsilon_scheduler: Scheduler | None = None,
        temperature_scheduler: Scheduler | None = None,
        replay_buffer: ReplayBuffer | None = None,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)
        super().__init__(
            forward_policy,
            action_embedder,
            optimizer,
            scheduler,
            epsilon_scheduler=epsilon_scheduler,
            temperature_scheduler=temperature_scheduler,
            replay_buffer=replay_buffer,
            **kwargs,
        )

        # Contrary to A2C, this one outputs a probability vector instead of one value.
        # This is the Q-value function Q(s,a) whereas in A2C it was the value function V(s).
        self.critic_1 = critic
        self.critic_2 = deepcopy(critic)
        self.critic_1_target = deepcopy(critic)
        self.critic_2_target = deepcopy(critic)

        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

    def configure_optimizers(self):
        params = []
        if self.forward_policy is not None and has_trainable_parameters(
            self.forward_policy
        ):
            params.append(
                {
                    "params": self.forward_policy.parameters(),
                    "lr": self.hparams.forward_policy_lr,
                }
            )
        if self.action_embedder is not None and has_trainable_parameters(
            self.action_embedder
        ):
            params.append(
                {
                    "params": self.action_embedder.parameters(),
                    "lr": self.hparams.action_embedder_lr,
                }
            )

        if self.critic_1 is not None and has_trainable_parameters(self.critic_1):
            params.append(
                {
                    "params": self.critic_1.parameters(),
                    "lr": self.hparams.critic_lr,
                }
            )

        if self.critic_2 is not None and has_trainable_parameters(self.critic_2):
            params.append(
                {
                    "params": self.critic_2.parameters(),
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

    def get_target_critic_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get the critic values for the current state.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): State.
        Return:
            q_values (torch.Tensor[batch_size, n_actions]): Forward logits.
        """
        processed = self.env.preprocess_states(state)
        if isinstance(processed, tuple):
            action_embedding_1 = self.critic_1_target(*processed)
            action_embedding_2 = self.critic_2_target(*processed)
        else:
            action_embedding_1 = self.critic_1_target(processed)
            action_embedding_2 = self.critic_2_target(processed)
        dim = action_embedding_1.shape[-1]
        library_embeddings = self.get_library_embeddings()
        q_values_1 = torch.einsum(
            "bd, nd -> bn", action_embedding_1, library_embeddings
        ) / (dim**0.5)  # Same as in softmax
        q_values_2 = torch.einsum(
            "bd, nd -> bn", action_embedding_2, library_embeddings
        ) / (dim**0.5)  # Same as in softmax
        return q_values_1, q_values_2

    def get_critic_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get the critic values for the current state.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): State.
        Return:
            q_values (torch.Tensor[batch_size, n_actions]): Forward logits.
        """
        processed = self.env.preprocess_states(state)
        if isinstance(processed, tuple):
            action_embedding_1 = self.critic_1(*processed)
            action_embedding_2 = self.critic_2(*processed)
        else:
            action_embedding_1 = self.critic_1(processed)
            action_embedding_2 = self.critic_2(processed)
        dim = action_embedding_1.shape[-1]
        library_embeddings = self.get_library_embeddings()
        q_values_1 = torch.einsum(
            "bd, nd -> bn", action_embedding_1, library_embeddings
        ) / (dim**0.5)  # Same as in softmax
        q_values_2 = torch.einsum(
            "bd, nd -> bn", action_embedding_2, library_embeddings
        ) / (dim**0.5)  # Same as in softmax
        return q_values_1, q_values_2

    def _compute_loss(self, trajectories, actions, dones, logreward):
        """Compute the loss for the model.
        Args:
            trajectories (torch.Tensor[batch_size, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            logreward (torch.Tensor[batch_size]): Log reward.
        """
        dones_ = rearrange(dones[:, :-1], "b t ... -> (b t) ...")
        current_states = rearrange(trajectories[:, :-1], "b t ... -> (b t) ...")  # s_t
        next_states = rearrange(trajectories[:, 1:], "b t ... -> (b t) ...")  # s_t+1
        current_actions = rearrange(actions, "b t ... -> (b t) ...").unsqueeze(1)  # a_t
        reward = torch.zeros(
            (trajectories.shape[0], trajectories.shape[1] - 1),
            device=logreward.device,
            dtype=logreward.dtype,
        )
        # Find first index where dones is True, the index before will contain the reward
        idx = dones.long().argmax(dim=-1) - 1
        reward[torch.arange(dones.shape[0]), idx] = (
            logreward.exp()
        )  # reward is zero everywhere except for final state
        reward = rearrange(reward, "b t ... -> (b t) ...")

        current_logits = self.get_forward_logits(current_states)  # pi(a|s_t)
        current_logits = torch.where(
            self.env.get_forward_mask(current_states),
            current_logits,
            torch.tensor(NEGATIVE_INFINITY).to(current_logits),
        )

        current_probs = Categorical(logits=current_logits).probs
        current_probs = torch.clip(current_probs, min=EPS)
        current_q_values_1, current_q_values_2 = self.get_critic_values(
            current_states
        )  # Q(s_t, a)

        with torch.no_grad():
            next_logits = self.get_forward_logits(next_states)  # pi(a|s_t+1)
            next_logits = torch.where(
                self.env.get_forward_mask(next_states),
                next_logits,
                torch.tensor(NEGATIVE_INFINITY).to(next_logits),
            )
            next_probs = torch.clip(Categorical(logits=next_logits).probs, min=EPS)

            next_q_target_1, next_q_target_2 = self.get_target_critic_values(
                next_states
            )  # Q(s_t+1, a)

            next_target_1 = (
                (next_q_target_1 - self.hparams.entropy_coefficient * next_probs.log())
                * next_probs
            ).sum(-1)  # V(s_t+1)

            next_value_1 = (
                reward
                + (1 - dones_.long()) * self.hparams.discount_factor * next_target_1
            )
            next_target_2 = (
                (next_q_target_2 - self.hparams.entropy_coefficient * next_probs.log())
                * next_probs
            ).sum(-1)  # V(s_t+1)
            next_value_2 = (
                reward
                + (1 - dones_.long()) * self.hparams.discount_factor * next_target_2
            )

        # Q-value loss
        q_loss_1 = 0.5 * (
            current_q_values_1.gather(1, current_actions.long()).squeeze()
            - next_value_1
        ).pow(2)
        q_loss_1 = q_loss_1[~dones_].mean()
        q_loss_2 = 0.5 * (
            current_q_values_2.gather(1, current_actions.long()).squeeze()
            - next_value_2
        ).pow(2)
        q_loss_2 = q_loss_2[~dones_].mean()
        q_loss = q_loss_1 + q_loss_2

        q_values = torch.min(current_q_values_1, current_q_values_2).detach()

        # Policy Loss
        policy_loss = (
            (self.hparams.entropy_coefficient * current_probs.log() - q_values)
            * current_probs
        ).mean(-1)
        policy_loss = policy_loss[~dones_].mean()

        loss = q_loss + policy_loss

        return loss, q_loss, policy_loss

    def compute_loss(self, trajectories, actions, dones, logreward):
        """Compute the loss for the model with chunking to avoid OOM errors.
        Args:
            trajectories (torch.Tensor[batch_size, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            logreward (torch.Tensor[batch_size]): Log reward.
        Returns:
            loss (torch.Tensor[1]): Total loss.
            q_loss (torch.Tensor[1]): Q-value loss.
            policy_loss (torch.Tensor[1]): Policy loss.
        """
        # Estimate the maximum chunk size
        max_chunk_size = self.hparams.max_chunk_size
        bs = trajectories.shape[0]
        num_chunks = (
            bs + max_chunk_size - 1
        ) // max_chunk_size  # Calculate number of chunks

        total_loss, total_q_loss, total_policy_loss = 0.0, 0.0, 0.0

        for i in range(num_chunks):
            start_idx = i * max_chunk_size
            end_idx = min(start_idx + max_chunk_size, bs)

            # Extract the chunk
            trajectories_chunk = trajectories[start_idx:end_idx]
            actions_chunk = actions[start_idx:end_idx]
            dones_chunk = dones[start_idx:end_idx]
            logreward_chunk = logreward[start_idx:end_idx]

            # Compute loss for the chunk
            loss, q_loss, policy_loss = self._compute_loss(
                trajectories_chunk, actions_chunk, dones_chunk, logreward_chunk
            )

            # Aggregate losses
            total_loss += loss * (end_idx - start_idx)
            total_q_loss += q_loss * (end_idx - start_idx)
            total_policy_loss += policy_loss * (end_idx - start_idx)

        # Average the losses across all chunks
        total_loss /= bs
        total_q_loss /= bs
        total_policy_loss /= bs

        return total_loss, total_q_loss, total_policy_loss

    def training_step(self, train_batch, batch_idx) -> Any:
        loss, q_loss, policy_loss = super().training_step(train_batch, batch_idx)

        # Update target networks
        if (
            self.current_epoch % self.hparams.target_network_frequency == 0
            and self.current_epoch > 0
        ):
            for param, target_param in zip(
                self.critic_1.parameters(), self.critic_1_target.parameters()
            ):
                target_param.data.copy_(
                    self.hparams.tau * param.data
                    + (1 - self.hparams.tau) * target_param.data
                )
            for param, target_param in zip(
                self.critic_2.parameters(), self.critic_2_target.parameters()
            ):
                target_param.data.copy_(
                    self.hparams.tau * param.data
                    + (1 - self.hparams.tau) * target_param.data
                )
        self.log(
            "train/q_loss",
            q_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/policy_loss",
            policy_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx) -> Any:
        final_state, logreward = val_batch
        x, trajectories, actions, dones, final_state, logreward, trajectory_length = (
            self.sample(final_state, train=False, epsilon=None, temperature=None)
        )
        loss = self.compute_loss(trajectories, actions, dones, logreward)

        self.val_loss(loss)
        self.val_logreward(logreward.mean())

        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)
        self.log(
            "val/logreward",
            self.val_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        # Get on-policy samples from the GFN
        dummy_batch = torch.arange(self.hparams.n_onpolicy_samples).to(self.device)
        _, _, actions, dones, final_state, _, trajectory_length = self.sample(
            dummy_batch,
            train=False,
            epsilon=None,
            temperature=None,
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
                self.env.action_frequency,
                self.env.actions,
            )
        ]
        torch.save(
            action_frequency, f"{self.trainer.log_dir}/library_{self.current_epoch}.pt"
        )
