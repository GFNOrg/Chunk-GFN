from typing import Any

import torch
from torch import nn
from torch.optim import Adam

from chunkgfn.algo.sampler_base import BaseSampler
from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer


class RandomSampler(BaseSampler):
    """Module for a random sampler. This baseline samples actions uniformly at random."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer | None = None,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)
        forward_policy = None
        action_embedder = None
        super().__init__(
            forward_policy,
            action_embedder,
            optimizer=None,
            scheduler=None,
            epsilon_scheduler=None,
            temperature_scheduler=None,
            replay_buffer=replay_buffer,
            **kwargs,
        )

    def configure_optimizers(self):
        params = [
            {
                "params": nn.Parameter(torch.zeros(1)),
                "lr": 0,
            }
        ]  # dummy params for lightning not to complain

        optimizer = Adam(params, lr=0.0)
        return {"optimizer": optimizer}

    def compute_loss(self, trajectories, actions, dones, logreward):
        """Dummy loss."""
        return None

    def get_forward_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get the forward logits for the given state.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): State.
        Return:
            logits (torch.Tensor[batch_size, n_actions]): Forward logits.
        """
        bs = state.shape[0]
        logits = torch.zeros(bs, self.env.n_actions).to(state.device)
        return logits

    def validation_step(self, val_batch, batch_idx) -> Any:
        final_state, _ = val_batch
        _, _, _, _, final_state, logreward, _ = self.sample(
            final_state,
        )

        self.val_logreward(logreward.mean())

        self.log(
            "val/logreward",
            self.val_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        # Get on-policy samples from the sampler
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
