from typing import Any

import torch
from einops import rearrange
from torch import nn
from torch.distributions import Categorical

from chunkgfn.algo.sampler_base import BaseSampler
from chunkgfn.algo.utils import has_trainable_parameters

from ..constants import NEGATIVE_INFINITY


class A2C(BaseSampler):
    """Advantage Actor Critic module."""

    def __init__(
        self,
        forward_policy: nn.Module,
        action_embedder: nn.Module,
        critic: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)
        super().__init__(
            forward_policy,
            action_embedder,
            optimizer,
            scheduler,
            epsilon_scheduler=None,
            temperature_scheduler=None,
            replay_buffer=None,
            **kwargs,
        )
        self.critic = critic

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

        if self.critic is not None and has_trainable_parameters(self.critic):
            params.append(
                {
                    "params": self.critic.parameters(),
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
        processed = self.env.preprocess_states(states)
        if isinstance(processed, tuple):
            values = self.critic(*processed).squeeze()
        else:
            values = self.critic(processed).squeeze()

        advantage = returns - values

        logits = self.get_forward_logits(states)
        forward_mask = self.env.get_forward_mask(states)
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
