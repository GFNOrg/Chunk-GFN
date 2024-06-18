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
        pass
