from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.distributions import Categorical
from torchmetrics import SpearmanCorrCoef

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
        self.save_hyperparameters(logger=False)
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
        self._logZ = nn.Parameter(
            torch.ones(self.hparams.num_partition_nodes)
            * self.hparams.partition_init
            / self.hparams.num_partition_nodes
        )

        self.val_correlation = SpearmanCorrCoef()

    @property
    def logZ(self):
        return self._logZ.sum()

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
        params.append(
            {
                "params": self._logZ,
                "lr": self.hparams.partition_lr,
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

    @torch.no_grad()
    def refactor_replay_buffer(self):
        """Refactor the replay buffer. This function takes final states from the replay
        buffer and samples backward trajectories for them to get different trajctories
        based on the current library.
        """
        if self.replay_buffer is not None:
            final_state = self.replay_buffer.storage["final_state"]
            trajectories, actions, dones, _ = self.go_backward(
                final_state.to(self.device)
            )
            self.replay_buffer.storage["trajectories"] = trajectories.cpu()
            self.replay_buffer.storage["actions"] = actions.cpu()
            self.replay_buffer.storage["dones"] = dones.cpu()

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
        device = final_state.device
        state = final_state.clone()
        done = torch.zeros((bs), device=device, dtype=bool)

        # Start unrolling the trajectories
        actions = []
        trajectories = []
        dones = []
        dones.append(torch.ones((bs), device=device, dtype=bool))
        while not done.all():
            logit_pb = self.get_backward_logits(state)
            backward_mask = self.env.get_backward_mask(state)
            logit_pb = torch.where(
                backward_mask,
                logit_pb,
                torch.tensor(NEGATIVE_INFINITY, device=device),
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

    @torch.no_grad()
    def get_ll(
        self, final_state: torch.Tensor, logreward: torch.Tensor
    ) -> torch.Tensor:
        """Get the log likelihood of the model for the given trajectories.
        Args:
            final_state (torch.Tensor[batch_size, *state_shape]): The examples for which we're computing the log-likelihood.
            logreward (torch.Tensor[batch_size]): Log reward.
        Return:
            log_pT (torch.Tensor[batch_size]): Log likelihood.
            trajectories (torch.Tensor[batch_size*n_trajectories, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size*n_trajectories, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size*n_trajectories, trajectory_length]): Whether the trajectory is done or not.
            logreward (torch.Tensor[batch_size*n_trajectories]): Log reward.
        """
        unique_final_state = final_state
        bs = unique_final_state.shape[0]
        device = final_state.device
        # Repeat the final_state n_trajectories times
        final_state = repeat(
            final_state, "b ... -> b n ...", n=self.hparams.n_trajectories
        )
        final_state = rearrange(final_state, "b n ... -> (b n) ...")
        logreward = repeat(logreward, "b -> b n", n=self.hparams.n_trajectories)
        logreward = rearrange(logreward, "b n ... -> (b n) ... ")
        trajectories, actions, dones, final_state = self.go_backward(final_state)

        # Calculate the log likelihood
        log_pf = 0
        log_pb = 0

        for t in range(trajectories.shape[1]):
            state = trajectories[:, t]
            logit_pf = self.get_forward_logits(state)
            forward_mask = self.env.get_forward_mask(state)
            logit_pf = torch.where(
                forward_mask,
                logit_pf,
                torch.tensor(NEGATIVE_INFINITY, device=device),
            )

            if t < trajectories.shape[1] - 1:
                log_pf += (Categorical(logits=logit_pf).log_prob(actions[:, t])) * (
                    ~dones[:, t] + 0
                )

            if t > 0:
                logit_pb = self.get_backward_logits(state)
                backward_mask = self.env.get_backward_mask(state)

                logit_pb = torch.where(
                    backward_mask,
                    logit_pb,
                    torch.tensor(NEGATIVE_INFINITY, device=device),
                )

                # When no action is available, just fill with uniform because
                # it won't be picked anyway in the backward_step.
                # Doing this avoids having nan when computing probabilities
                logit_pb = torch.where(
                    (logit_pb == -torch.inf).all(dim=-1).unsqueeze(1),
                    torch.tensor(0.0),
                    logit_pb,
                )
                log_pb += torch.where(
                    dones[:, t] | self.trainer.datamodule.is_initial_state(state),
                    torch.tensor(0.0),
                    Categorical(logits=logit_pb).log_prob(actions[:, t - 1]),
                )

        log_pb = rearrange(log_pb, "(b n) ... -> b n ...", b=bs).to(final_state.device)
        log_pf = rearrange(log_pf, "(b n) ... -> b n ...", b=bs)

        assert log_pf.shape == log_pb.shape
        log_pT = torch.logsumexp(log_pf - log_pb, dim=1) - torch.log(
            torch.tensor(self.hparams.n_trajectories)
        )

        return log_pT, trajectories, actions, dones, logreward

    def compute_loss(self, trajectories, actions, dones, logreward):
        """Compute the loss for the model.
        Args:
            trajectories (torch.Tensor[batch_size, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            logreward (torch.Tensor[batch_size]): Log reward.
        Return:
            loss (torch.Tensor[1]): Loss.
        """

        bs = trajectories.shape[0]
        device = trajectories.device
        trajectories_forward = rearrange(trajectories[:, :-1], "b t ... -> (b t) ...")
        dones_forward = rearrange(dones[:, :-1], "b t ... -> (b t) ...")
        actions_ = rearrange(actions, "b t ... -> (b t) ...")

        logit_pf = self.get_forward_logits(trajectories_forward)
        forward_mask = self.env.get_forward_mask(trajectories_forward)
        logit_pf = torch.where(
            forward_mask, logit_pf, torch.tensor(NEGATIVE_INFINITY, device=device)
        )

        log_pf_ = Categorical(logits=logit_pf).log_prob(actions_) * (~dones_forward + 0)
        log_pf = rearrange(log_pf_, "(b t) ... -> b t ...", b=bs).sum(1)

        trajectories_backward = rearrange(trajectories[:, 1:], "b t ... -> (b t) ...")

        dones_backward = rearrange(dones[:, 1:], "b t ... -> (b t) ...")

        backward_mask = self.env.get_backward_mask(trajectories_backward)

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

        return loss

    def training_step(self, train_batch, batch_idx) -> Any:
        loss = super().training_step(train_batch, batch_idx)
        self.log(
            "logZ",
            self.logZ,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx) -> Any:
        final_state, logreward = val_batch
        log_pT, trajectories, actions, dones, logreward = self.get_ll(
            final_state, logreward
        )
        loss = self.compute_loss(trajectories, actions, dones, logreward)

        logreward = rearrange(
            logreward, "(b n) ... -> b n ...", b=final_state.shape[0]
        ).mean(dim=1)

        self.val_loss(loss)
        self.val_logreward(logreward.mean())
        self.val_correlation(log_pT, logreward)

        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)
        self.log(
            "val/logreward",
            self.val_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/correlation",
            self.val_correlation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        # Get on-policy samples from the GFN
        dummy_batch = torch.arange(self.hparams.n_onpolicy_samples, device=self.device)
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
                self.trainer.datamodule.action_frequency,
                self.trainer.datamodule.actions,
            )
        ]
        torch.save(
            action_frequency, f"{self.trainer.log_dir}/library_{self.current_epoch}.pt"
        )
