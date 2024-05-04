from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.optim import lr_scheduler as lr_scheduler
from torch.optim.optimizer import Optimizer as Optimizer

from chunkgfn.gfn.base_unconditional_gfn import UnConditionalSequenceGFN
from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer
from chunkgfn.replay_buffer.utils import extend_trajectories
from chunkgfn.schedulers import Scheduler

from ..constants import NEGATIVE_INFINITY


class TBGFN_Variable(UnConditionalSequenceGFN):
    def __init__(
        self,
        forward_model: nn.Module,
        backward_model: nn.Module,
        action_model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        epsilon_scheduler: Scheduler | None = None,
        temperature_scheduler: Scheduler | None = None,
        replay_buffer: ReplayBuffer | None = None,
        **kwargs,
    ):
        super().__init__(
            forward_model,
            backward_model,
            action_model,
            optimizer,
            scheduler,
            epsilon_scheduler,
            temperature_scheduler,
            replay_buffer,
            **kwargs,
        )
        self.automatic_optimization = False

    def compute_loss(self, trajectories, actions, dones, logreward):
        """Compute the loss for the model.
        Args:
            trajectories (torch.Tensor[batch_size, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            logreward (torch.Tensor[batch_size]): Log reward.
        Return:
            loss (torch.Tensor[1]): Loss.
            logZ (torch.Tensor[1]): Log partition function.
        """
        log_pf = 0
        log_pb = 0
        for t in range(trajectories.shape[1]):
            state = trajectories[:, t]
            logit_pf = self.get_forward_logits(state)
            forward_mask = self.trainer.datamodule.get_forward_mask(state)
            logit_pf = torch.where(
                forward_mask,
                logit_pf,
                torch.tensor(NEGATIVE_INFINITY).to(logit_pf),
            )
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

        logZ = self.logZ

        loss = F.mse_loss(
            logZ + log_pf, (logreward / self.hparams.reward_temperature) + log_pb
        )

        return loss, logZ

    @torch.no_grad()
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
            self.update_library(n=self.hparams.n_chunks)

        else:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        return loss
