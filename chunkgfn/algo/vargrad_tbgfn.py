from typing import Any

import torch
from einops import rearrange
from torch import nn
from torch.distributions import Categorical

from chunkgfn.algo.tb_gfn import TBGFN
from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer
from chunkgfn.schedulers import Scheduler

from ..constants import NEGATIVE_INFINITY


class VarGradTBGFN(TBGFN):
    def __init__(
        self,
        forward_policy: nn.Module,
        action_embedder: nn.Module,
        backward_policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        logit_scaler: torch.nn.Module | None = None,
        epsilon_scheduler: Scheduler | None = None,
        temperature_scheduler: Scheduler | None = None,
        replay_buffer: ReplayBuffer | None = None,
        **kwargs,
    ):
        super().__init__(
            forward_policy,
            action_embedder,
            backward_policy,
            optimizer,
            scheduler,
            logit_scaler,
            epsilon_scheduler,
            temperature_scheduler,
            replay_buffer,
            **kwargs,
        )
        self.__logZ = 0

    @property
    def logZ(self):
        return self.__logZ

    def compute_loss(self, trajectories, actions, dones, logreward):
        """Compute the loss for the model with chunking to avoid OOM errors.
        Args:
            trajectories (torch.Tensor[batch_size, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            logreward (torch.Tensor[batch_size]): Log reward.
            max_chunk_size (int): Maximum chunk size to process at once.
        Return:
            loss (torch.Tensor[1]): Loss.
        """

        bs = trajectories.shape[0]
        device = trajectories.device
        max_chunk_size = self.hparams.max_chunk_size
        num_chunks = (
            bs + max_chunk_size - 1
        ) // max_chunk_size  # Calculate number of chunks

        total_loss = 0.0
        logZ = 0

        for i in range(num_chunks):
            # Define the chunk range
            start_idx = i * max_chunk_size
            end_idx = min(start_idx + max_chunk_size, bs)

            # Extract the chunk
            trajectories_chunk = trajectories[start_idx:end_idx]
            actions_chunk = actions[start_idx:end_idx]
            dones_chunk = dones[start_idx:end_idx]
            logreward_chunk = logreward[start_idx:end_idx]

            # Forward pass for the chunk
            trajectories_forward = rearrange(
                trajectories_chunk[:, :-1], "b t ... -> (b t) ..."
            )
            dones_forward = rearrange(dones_chunk[:, :-1], "b t ... -> (b t) ...")
            actions_ = rearrange(actions_chunk, "b t ... -> (b t) ...")

            logit_pf = self.get_forward_logits(trajectories_forward)
            forward_mask = self.env.get_forward_mask(trajectories_forward)
            logit_pf = torch.where(
                forward_mask, logit_pf, torch.tensor(NEGATIVE_INFINITY, device=device)
            )

            log_pf_ = Categorical(logits=logit_pf).log_prob(actions_) * (
                ~dones_forward + 0
            )
            log_pf = rearrange(
                log_pf_, "(b t) ... -> b t ...", b=logreward_chunk.size(0)
            ).sum(1)

            # Backward pass for the chunk
            trajectories_backward = rearrange(
                trajectories_chunk[:, 1:], "b t ... -> (b t) ..."
            )
            dones_backward = rearrange(dones_chunk[:, 1:], "b t ... -> (b t) ...")

            logit_pb = self.get_backward_logits(trajectories_backward)
            backward_mask = self.env.get_backward_mask(trajectories_backward)
            logit_pb = torch.where(
                backward_mask, logit_pb, torch.tensor(NEGATIVE_INFINITY, device=device)
            )

            log_pb_ = Categorical(logits=logit_pb).log_prob(actions_) * (
                ~dones_backward + 0
            )
            log_pb = rearrange(
                log_pb_, "(b t) ... -> b t ...", b=logreward_chunk.size(0)
            ).sum(1)

            # Calculate loss for the chunk
            logZ_ = (
                (logreward_chunk / self.hparams.reward_temperature) + log_pb
            ) - log_pf
            chunk_loss = torch.var(logZ_)

            # Aggregate the loss
            total_loss += chunk_loss * (end_idx - start_idx)
            logZ += logZ_.mean() * (end_idx - start_idx)

        # Average the loss across all chunks
        total_loss /= bs
        logZ /= bs
        self.__logZ = logZ

        return total_loss
