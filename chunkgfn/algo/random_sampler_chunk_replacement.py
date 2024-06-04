from typing import Any

import torch

from chunkgfn.algo.random_sampler import RandomSampler
from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer
from chunkgfn.replay_buffer.utils import extend_trajectories


class RandomSamplerChunkReplacement(RandomSampler):
    """Abstract class for sequence-based Generative Flow Networks."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer | None = None,
        **kwargs,
    ):
        super().__init__(
            optimizer,
            replay_buffer,
            **kwargs,
        )

    def update_library(self):
        """Update the library. This function will do the following, in the following order:
        1. Pick a number of generated samples from the replay buffer.
        2. Transform samples into their usual data structure.
        3. Apply a tokenizing algorithm to get the most valuable token.
        4. Update the logits_layer to reflect the added token.
        """

        # Pick a number of generated samples from the replay buffer
        nsamples_replay = int(
            self.hparams.n_samples * self.hparams.ratio_from_replay_buffer
        )

        samples = self.replay_buffer.sample(nsamples_replay)
        trajectories_rb = samples["trajectories"]
        actions_rb = samples["actions"]
        dones_rb = samples["dones"]
        trajectories, actions, dones, _, _ = self.forward(
            self.hparams.n_samples - nsamples_replay
        )
        # Concatenate samples from the replay buffer and the on-policy samples
        _, actions, dones = extend_trajectories(
            trajectories.to(trajectories_rb),
            trajectories_rb,
            actions.to(actions_rb),
            actions_rb,
            dones.to(dones_rb),
            dones_rb,
        )

        n = self.hparams.total_library_size - len(self.trainer.datamodule.atomic_tokens)
        if self.hparams.chunk_algorithm == "bpe":
            self.trainer.datamodule.chunk_bpe(
                samples["actions"], samples["dones"], n_tokens_to_add=n, remove_old=True
            )
        elif self.hparams.chunk_algorithm == "wordpiece":
            self.trainer.datamodule.chunk_wordpiece(
                samples["actions"], samples["dones"], n_tokens_to_add=n, remove_old=True
            )
        elif self.hparams.chunk_algorithm == "uniform":
            self.trainer.datamodule.chunk_uniform(n_tokens_to_add=n, remove_old=True)
        else:
            raise Exception("chunk_algorithm not in ['bpe', 'wordpiece', 'uniform']")

    def training_step(self, train_batch, batch_idx) -> Any:
        loss = super().training_step(train_batch, batch_idx)

        if (
            self.current_epoch > 0
            and self.current_epoch % self.hparams.library_update_frequency == 0
            and batch_idx == 0
        ):
            self.update_library()
            self.refactor_replay_buffer()
        return loss
