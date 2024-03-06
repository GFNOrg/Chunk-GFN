from typing import Any

import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.distributions import Categorical
from torch.optim import lr_scheduler as lr_scheduler
from torch.optim.optimizer import Optimizer as Optimizer

from chunkgfn.gfn.base_unconditional_gfn import UnConditionalSequenceGFN
from chunkgfn.models.utils import expand_linear_layer
from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer
from chunkgfn.schedulers import Scheduler


class TBGFN_Variable(UnConditionalSequenceGFN):
    def __init__(
        self,
        forward_model: nn.Module,
        backward_model: nn.Module,
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
            trajectories (torch.Tensor[batch_size, trajectory_length, seq_length, state_dim]): Trajectories for each sample in the batch.
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
            logp_f_s = self.forward_model(state)
            if t < trajectories.shape[1] - 1:
                log_pf += (Categorical(logits=logp_f_s).log_prob(actions[:, t])) * (
                    ~dones[:, t] + 0
                )
            if t > 0:
                backward_actions = self.trainer.datamodule.get_parent_actions(state)
                logp_b_s = torch.where(
                    backward_actions == 1, torch.tensor(0.0), -torch.inf
                ).to(logp_f_s)
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

    def update_library(self):
        """Update the library. This function will do the following, in the following order:
        1. Pick a number of generated samples from the replay buffer.
        2. Transform samples into their usual data structure.
        3. Apply a tokenizing algorithm to get the most valuable token.
        4. Update the logits_layer to reflect the added token.
        """

        # Pick a number of generated samples from the replay buffer
        samples = self.replay_buffer.sample(self.hparams.n_samples)
        # Get the most valuable token
        self.trainer.datamodule.chunk(samples["final_state"])

        # Update model's weights
        def init_weights(m):
            m.bias.data.fill_(0.0)
            m.weight.data.fill_(0.0)

        self.forward_model.logits_layer = expand_linear_layer(
            self.forward_model.logits_layer,
            new_out_dim=self.forward_model.logits_layer.out_features + 1,
            init_weights=init_weights,
        )

        # Reinitialize the optimizer
        self.trainer.optimizers = [self.configure_optimizers()["optimizer"]]

    def training_step(self, train_batch, batch_idx) -> Any:
        if self.epsilon_scheduler is not None:
            epsilon = self.epsilon_scheduler.step(self.current_epoch)
        else:
            epsilon = None
        if self.temperature_scheduler is not None:
            temperature = self.temperature_scheduler.step(self.current_epoch)
        else:
            temperature = None

        x, trajectories, actions, dones, final_state, logreward = self.sample(
            train_batch,
            train=True,
            epsilon=epsilon,
            temperature=temperature,
        )
        batch_size = x.shape[0]
        nsamples_replay = int(batch_size * self.hparams.ratio_from_replay_buffer)

        if self.replay_buffer is not None:
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
            trajectories = torch.cat(
                [trajectories[indices], samples["trajectories"]], dim=0
            )
            actions = torch.cat([actions[indices], samples["actions"]], dim=0)
            dones = torch.cat([dones[indices], samples["dones"]], dim=0)
            final_state = torch.cat(
                [final_state[indices], samples["final_state"]], dim=0
            )
            logreward = torch.cat([logreward[indices], samples["logreward"]], dim=0)

        loss, logZ = self.compute_loss(trajectories, actions, dones, logreward)
        additional_metrics = self.trainer.datamodule.compute_metrics(final_state)

        self.train_loss(loss)
        self.train_logreward(logreward.mean())
        self.train_logZ(logZ.mean())

        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)
        self.log(
            "train/logreward",
            self.train_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/logZ",
            self.train_logZ,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.epsilon_scheduler is not None:
            self.log("epsilon", epsilon, on_step=False, on_epoch=True, prog_bar=False)
        if self.temperature_scheduler is not None:
            self.log(
                "temperature", epsilon, on_step=False, on_epoch=True, prog_bar=False
            )

        self.log(
            "replay_buffer_size",
            len(self.replay_buffer),
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
        for metric_name in additional_metrics:
            self.log(
                f"train/{metric_name}",
                additional_metrics[metric_name],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if (
            self.current_epoch > 0
            and self.current_epoch % self.hparams.library_update_frequency == 0
            and batch_idx == 0
        ):
            self.update_library()

        else:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        return loss

    def on_train_epoch_end(self) -> None:
        rows = []
        rows.append(
            [
                "|".join(self.trainer.datamodule.actions),
            ]
        )
        self.logger.experiment.log(
            {
                "text_samples": wandb.Table(
                    columns=[
                        "library",
                    ],
                    data=rows,
                )
            }
        )
