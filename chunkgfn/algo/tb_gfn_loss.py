from typing import Any

import torch
from torch import nn

from chunkgfn.algo.tb_gfn import TBGFN
from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer
from chunkgfn.replay_buffer.utils import extend_trajectories
from chunkgfn.schedulers import Scheduler


class TBGFNLoss(TBGFN):
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

        assert (
            self.hparams.loss_multiplier < 1
        ), "loss_multiplier must be smaller than 1."
        self.loss_threshold = self.hparams.initial_loss_threshold
        self.loss_multiplier = self.hparams.loss_multiplier

    def training_step(self, train_batch, batch_idx) -> Any:
        if self.epsilon_scheduler is not None:
            epsilon = self.epsilon_scheduler.step(self.current_epoch)
        else:
            epsilon = None
        if self.temperature_scheduler is not None:
            temperature = self.temperature_scheduler.step(self.current_epoch)
        else:
            temperature = None

        x, trajectories, actions, dones, final_state, logreward, trajectory_length = (
            self.sample(
                train_batch,
                train=True,
                epsilon=epsilon,
                temperature=temperature,
            )
        )
        batch_size = x.shape[0]

        sampler_logreward = logreward

        if self.replay_buffer is not None:
            nsamples_replay = int(batch_size * self.hparams.ratio_from_replay_buffer)
            with torch.no_grad():
                self.replay_buffer.add(
                    input=x,
                    trajectories=trajectories,
                    actions=actions,
                    dones=dones,
                    final_state=final_state,
                    logreward=logreward,
                )
                samples = self.replay_buffer.sample(nsamples_replay)

            for key in samples.keys():
                samples[key] = samples[key].to(x.device)

            # Concatenate samples from the replay buffer and the on-policy samples
            indices = torch.randperm(len(x))[: batch_size - nsamples_replay]
            x = torch.cat([x[indices], samples["input"]], dim=0)
            trajectories, actions, dones = extend_trajectories(
                trajectories[indices],
                samples["trajectories"],
                actions[indices],
                samples["actions"],
                dones[indices],
                samples["dones"],
            )

            final_state = torch.cat(
                [final_state[indices], samples["final_state"]], dim=0
            )

            sampler_logreward = logreward[indices]
            logreward = torch.cat([sampler_logreward, samples["logreward"]], dim=0)

        loss = self.compute_loss(trajectories, actions, dones, logreward)
        additional_metrics = self.env.compute_metrics(final_state, logreward)

        if loss is not None:
            self.train_loss(loss)
        self.train_logreward(sampler_logreward.mean())
        self.train_trajectory_length(trajectory_length.float().mean())

        if loss is not None:
            self.log(
                "train/loss",
                self.train_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        self.log(
            "train/logreward",
            self.train_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.replay_buffer is not None:
            self.log(
                "replay_buffer_size",
                float(len(self.replay_buffer)),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "replay_buffer_mean_logreward",
                self.replay_buffer.storage["logreward"].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.log(
            "train/trajectory_length",
            self.train_trajectory_length,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "logZ",
            self.logZ,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        for metric_name in additional_metrics:
            self.log(
                f"train/{metric_name}",
                additional_metrics[metric_name],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.hparams.chunk_algorithm is not None:
            current_loss_value = self.train_loss.compute()
            if current_loss_value <= self.loss_threshold:
                self.update_library()
                if (
                    self.hparams.chunk_type == "replacement"
                    and self.replay_buffer is not None
                ):
                    self.refactor_replay_buffer()

                self.loss_threshold *= self.loss_multiplier
        return loss
