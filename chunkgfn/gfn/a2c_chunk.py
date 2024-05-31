from typing import Any

import torch

from chunkgfn.gfn.a2c import A2C
from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer
from chunkgfn.replay_buffer.utils import extend_trajectories
from chunkgfn.schedulers import Scheduler


class A2CChunk(A2C):
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
        super().__init__(
            forward_model,
            critic_model,
            action_model,
            optimizer,
            scheduler,
            epsilon_scheduler,
            temperature_scheduler,
            replay_buffer,
            **kwargs,
        )
        self.automatic_optimization = False

    def update_library(self, n):
        """Update the library. This function will do the following, in the following order:
        1. Pick a number of generated samples from the replay buffer.
        2. Transform samples into their usual data structure.
        3. Apply a tokenizing algorithm to get the most valuable token.
        4. Update the logits_layer to reflect the added token.

        n: number of tokens to add.
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
            self.hparams.n_samples - nsamples_replay, train=False
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

        # Get the most valuable token TODO: make n_tokens_to_add configurable.
        if self.hparams.chunk_algorithm == "bpe":
            self.trainer.datamodule.chunk_bpe(
                actions,
                dones,
                n_tokens_to_add=n,
            )
        elif self.hparams.chunk_algorithm == "wordpiece":
            self.trainer.datamodule.chunk_wordpiece(
                actions,
                dones,
                n_tokens_to_add=n,
            )
        elif self.hparams.chunk_algorithm == "uniform":
            self.trainer.datamodule.chunk_uniform(n_tokens_to_add=n)
        else:
            raise Exception("chunk_algorithm not in ['bpe', 'wordpiece', 'uniform']")

    def training_step(self, train_batch, batch_idx) -> Any:
        loss = super().training_step(train_batch, batch_idx)

        if (
            self.current_epoch > 0
            and self.current_epoch % self.hparams.library_update_frequency == 0
            and batch_idx == 0
        ):
            self.update_library(n=self.hparams.n_chunks)

        else:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        return loss
