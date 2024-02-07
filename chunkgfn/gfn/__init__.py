from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
import wandb
from einops import rearrange, repeat
from lightning import LightningModule
from torch import nn
from torch.distributions import Categorical
from torchmetrics import MeanMetric

from chunkgfn.gfn.utils import has_trainable_parameters
from chunkgfn.replay_buffer import ReplayBuffer
from chunkgfn.schedulers import Scheduler

NEG_INF = -1e6  # Negative infinity
SMALL_VALUE = 1e-6  # Small value to avoid division by zero


class SequenceGFN(ABC, LightningModule):
    """Abstract class for sequence-based Generative Flow Networks."""

    def __init__(
        self,
        forward_model: torch.nn.Module,
        backward_model: torch.nn.Module,
        partition_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        epsilon_scheduler: Scheduler | None = None,
        temperature_scheduler: Scheduler | None = None,
        replay_buffer: ReplayBuffer | None = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # GFlowNet modules
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.partition_model = partition_model

        # Schedulers for off-policy training
        self.epsilon_scheduler = epsilon_scheduler
        self.temperature_scheduler = temperature_scheduler

        self.replay_buffer = replay_buffer

        # Metric managers
        self.metric_manager = {
            "train": {
                "loss": MeanMetric(),
                "logreward": MeanMetric(),
                "logZ": MeanMetric(),
            },
            "val": {
                "loss": MeanMetric(),
                "logreward": MeanMetric(),
                "logZ": MeanMetric(),
            },
            "test": {
                "loss": MeanMetric(),
                "logreward": MeanMetric(),
                "logZ": MeanMetric(),
            },
        }

    def configure_optimizers(self):
        params = []
        for model in [self.forward_model, self.backward_model, self.partition_model]:
            if model is not None and has_trainable_parameters(model):
                params.append({"params": model.parameters(), "lr": self.hparams.lr})

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

    def update_state(
        self, state: torch.tensor, action: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        """Update the given state with the given action.
        We look for the first occurence of the padding element and replace it with the action.
        Args:
            state (torch.Tensor[batch_size, seq_length, state_dim]): State tensor.
            action (torch.Tensor[batch_size, n_actions]): Action tensor.
            done (torch.Tensor[batch_size]): Whether the trajectory is done or not.
        Return:
            new_state (torch.Tensor[batch_size, seq_length, state_dim]): Updated state tensor.
        """
        new_state = state.clone()

    def forward(
        self,
        x: torch.Tensor,
        train: bool = True,
        epsilon: float | None = None,
        temperature: float | None = None,
    ):
        """Sample forward trajectories conditioned on inputs.
        Args:
            x (torch.Tensor[batch_size, seq_length, input_dim]): Conditioning vector.
            train (bool): Whether it's during train or eval. This makes sure that we don't sample off-policy during inference.
            epsilon (float|None): Epsilon value for epsilon greedy.
            temperature (float|None): Temperature value for tempering.
        Return:
            trajectories (torch.Tensor[batch_size, trajectory_length, seq_length, state_dim]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length, state_dim]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            state (torch.Tensor[batch_size, seq_length, state_dim]): Final state.
        """
        bs, max_len, dim = x.shape
        eos_token_idx = self.trainer.datamodule.data_train.eos_token_idx
        state = -torch.ones(bs, max_len, dim).to(x)
        eos_token = torch.zeros(dim).to(x)
        eos_token[eos_token_idx] = 1
        done = torch.zeros((bs)).to(x).bool()

        # Start unrolling the trajectories
        actions = []
        trajectories = []
        dones = []

        for t in range(max_len):
            p_f_s = self.forward_model(x, state)
            uniform_dist_probs = torch.ones_like(p_f_s).to(p_f_s)

            valid_actions_mask = self.trainer.datamodule.get_invalid_actions_mask(state)
            p_f_s = torch.where(valid_actions_mask, p_f_s, torch.tensor(NEG_INF))
            uniform_dist_probs = torch.where(
                valid_actions_mask, uniform_dist_probs, torch.tensor(0)
            )

            if train:
                if temperature is not None:
                    logits = p_f_s / (SMALL_VALUE + temperature)
                else:
                    logits = p_f_s
                if epsilon is not None:
                    probs = torch.softmax(logits, dim=-1)
                    uniform_dist_probs = uniform_dist_probs / uniform_dist_probs.sum(
                        dim=-1, keepdim=True
                    )
                    probs = (1 - epsilon) * probs + epsilon * uniform_dist_probs
                    cat = Categorical(probs=probs)
                else:
                    cat = Categorical(logits=logits)
            else:
                cat = Categorical(logits=p_f_s)

            act = cat.sample()

            action = torch.zeros((bs, dim)).to(state)
            action.scatter_(-1, act.unsqueeze(-1), 1)

            # Update the state by filling the current timestep with the sampled action only if it doesn't contain EOS token
            new_state = state.clone()
            if t > 0:
                done |= torch.all(state[:, t - 1] == eos_token.unsqueeze(0), dim=1)
                new_state[:, t] = torch.where(done.unsqueeze(1), state[:, t], action)
            else:
                new_state[:, t] = action

            actions.append(action)
            trajectories.append(state)
            dones.append(done.clone())

            state = new_state.clone()

        trajectories.append(state)
        trajectories = torch.stack(trajectories, dim=1)
        actions = torch.stack(actions, dim=1)
        dones = torch.stack(dones, dim=1)

        return trajectories, actions, dones, state

    def compute_loss(self, x, trajectories, actions, dones, logreward):
        """Compute the loss for the model.
        Args:
            x (torch.Tensor[batch_size, seq_length, input_dim]): Conditioning vector.
            trajectories (torch.Tensor[batch_size, trajectory_length, seq_length, state_dim]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length, state_dim]): Actions for each sample in the batch.
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
            logp_f_s = self.forward_model(x, state)

            act = torch.argmax(actions[:, t], dim=-1)
            log_pf += (Categorical(logits=logp_f_s).log_prob(act)) * (~dones[:, t] + 0)
            if t > 0:
                backward_actions = self.trainer.datamodule.get_parent_actions(state)
                logp_b_s = torch.where(
                    backward_actions == 1, torch.tensor(0.0), -torch.inf
                ).to(logp_f_s)
                act = torch.argmax(actions[:, t - 1], dim=-1)
                log_pb += torch.where(
                    dones[:, t],
                    torch.tensor(0.0),
                    Categorical(logits=logp_b_s).log_prob(act),
                )

        logZ = self.partition_model(x).squeeze(-1)
        loss = self.mse_loss(logZ + log_pf, logreward + log_pb)

        return loss, logZ

    def sample(
        self,
        batch: torch.Tensor,
        eos_token_idx: int,
        state_vocab_size: int,
        train: bool = True,
        epsilon: float = 0.0,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch
        # Repeat the input n_trajectories times
        x = repeat(x, "b ... -> b n ...", n=self.hparams.n_trajectories)
        x = rearrange(x, "b n ... -> (b n) ...")
        trajectories, actions, dones, final_state = self(
            x, eos_token_idx, state_vocab_size, train, epsilon, temperature
        )
        logreward = (
            self.trainer.datamodule.compute_logreward(x, final_state).to(final_state)
            / self.hparams.reward_temperature
        )
        return x, trajectories, actions, dones, final_state, logreward

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

        loss, logZ = self.compute_loss(x, trajectories, actions, dones, logreward)
        accuracy = self.trainer.datamodule.compute_accuracy(x, final_state)

        self.train_loss(loss)
        self.train_logreward(logreward.mean())
        self.train_logZ(logZ.mean())
        self.train_accuracy(accuracy.mean())

        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)
        self.log(
            "train/logreward",
            self.train_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/accuracy",
            self.train_accuracy,
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

    def validation_step(self, val_batch, batch_idx) -> Any:
        x, trajectories, actions, dones, final_state, logreward = self.sample(
            val_batch,
            train=False,
            epsilon=None,
            temperature=None,
        )

        loss, logZ = self.compute_loss(x, trajectories, actions, dones, logreward)
        accuracy = self.trainer.datamodule.compute_accuracy(x, final_state)

        self.val_loss(loss)
        self.val_logreward(logreward.mean())
        self.val_logZ(logZ.mean())
        self.val_accuracy(accuracy.mean())

        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)
        self.log(
            "val/logreward",
            self.val_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/logZ",
            self.val_logZ,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
