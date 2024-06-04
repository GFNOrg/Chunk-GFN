import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.distributions import Categorical

from chunkgfn.algo.sampler_base import BaseSampler
from chunkgfn.algo.utils import has_trainable_parameters
from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer
from chunkgfn.schedulers import Scheduler

from ..constants import NEGATIVE_INFINITY


class TBGFN(BaseSampler):
    def __init__(
        self,
        forward_policy: nn.Module,
        action_embedder: nn.Module,
        backward_policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        epsilon_scheduler: Scheduler | None = None,
        temperature_scheduler: Scheduler | None = None,
        replay_buffer: ReplayBuffer | None = None,
        **kwargs,
    ):
        super().__init__(
            forward_policy,
            action_embedder,
            optimizer,
            scheduler,
            epsilon_scheduler,
            temperature_scheduler,
            replay_buffer,
            **kwargs,
        )
        self.backward_policy = backward_policy

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

        if self.backward_policy is not None and has_trainable_parameters(
            self.backward_policy
        ):
            params.append(
                {
                    "params": self.backward_policy.parameters(),
                    "lr": self.hparams.backward_policy_lr,
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

    def get_backward_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get the forward logits for the given state.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): State.
        Return:
            logits (torch.Tensor[batch_size, n_actions]): Forward logits.
        """
        action_embedding = self.backward_policy(self.env.preprocess_states(state))
        dim = action_embedding.shape[-1]
        library_embeddings = self.get_library_embeddings()
        logits = torch.einsum("bd, nd -> bn", action_embedding, library_embeddings) / (
            dim**0.5
        )  # Same as in softmax
        return logits

    def go_backward(self, final_state: torch.Tensor):
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
            logit_pb = self.get_backward_logits(state)
            backward_mask = self.env.get_backward_mask(state)

            logit_pb = torch.where(
                backward_mask,
                logit_pb,
                torch.tensor(NEGATIVE_INFINITY).to(logit_pb),
            )

            # When no action is available, just fill with uniform because
            # it won't be picked anyway in the backward_step.
            # Doing this avoids having nan when computing probabilities
            logit_pb = torch.where(
                (logit_pb == -torch.inf).all(dim=-1).unsqueeze(1),
                torch.tensor(0.0),
                logit_pb,
            )
            cat = Categorical(logits=logit_pb)

            act = cat.sample()

            new_state, done = self.env.backward_step(state, act)

            actions.append(act)
            trajectories.append(state)
            dones.append(done.clone())

            state = new_state.clone()

        trajectories.append(state)

        trajectories = torch.stack(trajectories[::-1], dim=1)
        actions = torch.stack(actions[::-1], dim=1)
        dones = torch.stack(dones[::-1], dim=1)

        return trajectories, actions, dones, final_state

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

        bs = trajectories.shape[0]
        trajectories_forward = rearrange(trajectories[:, :-1], "b t ... -> (b t) ...")
        dones_forward = rearrange(dones[:, :-1], "b t ... -> (b t) ...")
        actions_ = rearrange(actions, "b t ... -> (b t) ...")

        logit_pf = self.get_forward_logits(trajectories_forward)
        forward_mask = self.env.get_forward_mask(trajectories_forward)
        logit_pf = torch.where(
            forward_mask, logit_pf, torch.tensor(NEGATIVE_INFINITY).to(logit_pf)
        )

        log_pf_ = Categorical(logits=logit_pf).log_prob(actions_) * (~dones_forward + 0)
        log_pf = rearrange(log_pf_, "(b t) ... -> b t ...", b=bs).sum(1)

        trajectories_backward = rearrange(trajectories[:, 1:], "b t ... -> (b t) ...")

        dones_backward = rearrange(dones[:, 1:], "b t ... -> (b t) ...")

        backward_mask = self.env.get_parent_actions(trajectories_backward)

        logit_pb = torch.where(backward_mask == 1, torch.tensor(0.0), -torch.inf).to(
            logit_pf
        )

        logit_pb = torch.where(
            (logit_pb == -torch.inf).all(dim=-1).unsqueeze(1),
            torch.tensor(0.0),
            logit_pb,
        )
        log_pb_ = torch.where(
            dones_backward | self.env.is_initial_state(trajectories_backward),
            torch.tensor(0.0),
            Categorical(logits=logit_pb).log_prob(actions_),
        )
        log_pb = rearrange(log_pb_, "(b t) ... -> b t ...", b=bs).sum(1)

        logZ = self.logZ

        loss = F.mse_loss(
            logZ + log_pf, (logreward / self.hparams.reward_temperature) + log_pb
        )

        return loss, logZ
