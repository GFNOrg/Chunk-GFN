from typing import Any, Tuple

import torch
import wandb
from lightning import LightningModule
from torch import nn
from torch.distributions import Categorical
from torchmetrics import MeanMetric

from chunkgfn.models.utils import expand_linear_layer
from chunkgfn.replay_buffer import ReplayBuffer
from chunkgfn.schedulers import Scheduler


class Cond_TBGFN_Variable(LightningModule):
    def __init__(
        self,
        forward_model: torch.nn.Module,
        partition_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        epsilon_scheduler: Scheduler | None = None,
        temperature_scheduler: Scheduler | None = None,
        replay_buffer: ReplayBuffer | None = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.forward_model = forward_model
        self.logF_model = partition_model
        self.criterion = criterion
        self.epsilon_scheduler = epsilon_scheduler
        self.temperature_scheduler = temperature_scheduler
        self.replay_buffer = replay_buffer

        self.mse_loss = nn.MSELoss()

        self.train_loss = MeanMetric()
        self.train_logreward = MeanMetric()
        self.train_logZ = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_logreward = MeanMetric()
        self.val_logZ = MeanMetric()

    def configure_optimizers(self):
        params = [
            {
                "params": self.forward_model.parameters(),
                "lr": self.hparams.lr,
            },
            {
                "params": self.partition_model.parameters(),
                "lr": self.hparams.partition_lr,
            },
        ]
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

    def forward(
        self,
        x: torch.Tensor,
        eos_token_idx: int,
        state_vocab_size: int,
        train: bool = True,
        epsilon: float | None = None,
        temperature: float | None = None,
    ):
        """Sample trajectories conditioned on inputs.
        Args:
            x (torch.Tensor[batch_size, seq_length, input_dim]): Conditioning vector.
            eos_token_idx (int): Index of the EOS token in the vocabulary.
            state_vocab_size (int): Size of the state vocabulary.
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
        state = -torch.ones(bs, max_len, state_vocab_size).to(x)
        eos_token = torch.zeros(state_vocab_size).to(x)
        eos_token[eos_token_idx] = 1
        done = torch.zeros((bs)).to(x).bool()

        # Start unrolling the trajectories
        actions = []
        trajectories = []
        dones = []

        for t in range(max_len):
            p_f_s = self.forward_model(x, state)
            uniform_dist_probs = torch.ones_like(p_f_s).to(p_f_s)

            if t == max_len - 1:
                # If we're at the last timestep, we need to make sure that we sample ONLY the EOS token
                p_f_s[..., :eos_token_idx] = -1e6
                p_f_s[..., eos_token_idx + 1 :] = -1e6
                uniform_dist_probs[..., :eos_token_idx] = 0
                uniform_dist_probs[..., eos_token_idx + 1 :] = 0

            if train:
                if temperature is not None:
                    logits = p_f_s / (1e-6 + temperature)
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

            state = new_state

        trajectories = torch.stack(trajectories, dim=1)
        actions = torch.stack(actions, dim=1)
        dones = torch.stack(dones, dim=1)

        return trajectories, actions, dones, state

    # reward_start = imgs[j].log_prob(latent_disc.argmax(1)).sum((1, 2))
    #             reward_end = (
    #                 imgs[j + 1].log_prob(latent_disc.argmax(1)).sum((1, 2)) + prior_logprobs[j + 1]
    #             )

    #             energy_diff = reward_end - reward_start

    #             fw_loss += (
    #                 logF[j].view(batch_size, 1)
    #                 + logprobs_fw[j].view(batch_size, 1)
    #                 - logF[j + 1].view(batch_size, 1)
    #                 - energy_diff.view(batch_size, 1)
    #             ) ** 2

    def compute_loss(self, x, trajectories, actions, dones, logreward):
        log_pf = 0
        log_pb = 0

        fw_loss = 0

        for t in range(trajectories.shape[1] - 1):
            state = trajectories[:, t]
            p_f_s = self.forward_model(x, state)
            act = torch.argmax(actions[:, t], dim=-1)
            log_pf += (Categorical(logits=p_f_s).log_prob(act)) * (~dones[:, t] + 0)
            log_f_next = self.logF_model(x, state)
            if t > 0:
                log_f_parent = self.logF_model(x, trajectories[:, t - 1])
            else:
                log_f_parent = self.logF_model(x, torch.zeros_like(state))  ## should be logZ
                logZ = log_f_parent.squeeze(-1)
            logreward_end = self.trainer.datamodule.compute_logreward(x, trajectories[:, t])
            if t > 0:
                logreward_start = self.trainer.datamodule.compute_logreward(
                    x, trajectories[:, t - 1]
                )
            else:
                logreward_start = 0

            energy_diff = logreward_end - logreward_start
            fw_loss += (log_f_parent + log_pf - log_f_next - energy_diff) ** 2
        ##final step  take reward from logreward
        state = trajectories[:, -1]
        p_f_s = self.forward_model(x, state)
        act = torch.argmax(actions[:, -1], dim=-1)
        log_pf += (Categorical(logits=p_f_s).log_prob(act)) * (~dones[:, t] + 0)
        log_f_next = torch.zeros_like(log_f_next)  ## f_tilde = r
        log_f_parent = self.logF_model(x, trajectories[:, -2])
        logreward_end = logreward  ##final reward
        logreward_start = self.trainer.datamodule.compute_logreward(x, trajectories[:, -2])
        energy_diff = logreward_end - logreward_start
        fw_loss += (log_f_parent + log_pf - log_f_next - energy_diff) ** 2

        loss = fw_loss.mean() / trajectories.shape[1]
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
        trajectories, actions, dones, final_state = self(
            x, eos_token_idx, state_vocab_size, train, epsilon, temperature
        )
        logreward = self.trainer.datamodule.compute_logreward(x, final_state).to(final_state)
        return x, trajectories, actions, dones, final_state, logreward

    def update_library(self):
        """Update the library. This function will do the following, in the following order:
        1. Pick a number of generated samples from the replay buffer.
        2. Transform samples into their usual data structure.
        3. Apply a tokenizing algorithm to get the most valuable token.
        4. Update the forward_model's input_layer and logits_layer to reflect the added token.
        """

        # Pick a number of generated samples from the replay buffer
        _, final_states = self.replay_buffer.sample(self.hparams.n_samples)
        # Decode the samples
        final_states = torch.argmax(final_states, dim=-1)
        # Get the most valuable token

        # Update model's weights
        self.forward_model.input_embedding = expand_linear_layer(
            self.forward_model.state_embedding,
            new_in_dim=self.forward_model.state_embedding.in_features + 1,
        )
        self.forward_model.input_embedding = expand_linear_layer(
            self.forward_model.logits_layer,
            new_out_dim=self.forward_model.logits_layer.out_features + 1,
        )

    def training_step(self, train_batch, batch_idx) -> Any:
        eos_token_idx = self.trainer.train_dataloader.dataset.eos_token_idx
        state_vocab_size = self.trainer.train_dataloader.dataset.vocab_size

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
            eos_token_idx,
            state_vocab_size,
            train=True,
            epsilon=epsilon,
            temperature=temperature,
        )
        if self.replay_buffer is not None:
            with torch.no_grad():
                self.replay_buffer.add(x, trajectories, actions, dones, final_state, logreward)
                (
                    x,
                    trajectories,
                    actions,
                    dones,
                    final_state,
                    logreward,
                ) = self.replay_buffer.sample(x.shape[0])
        loss, logZ = self.compute_loss(x, trajectories, actions, dones, logreward)

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
            self.log("temperature", epsilon, on_step=False, on_epoch=True, prog_bar=False)

        generated_samples = self.trainer.datamodule.batch_token2vocab(final_state)
        input_samples = self.trainer.datamodule.batch_token2vocab(x)

        rows = []
        for i, (input_sample, generated_sample) in enumerate(zip(input_samples, generated_samples)):
            rows.append([input_sample, generated_sample, logreward[i].item()])
        self.logger.experiment.log(
            {
                "text_samples": wandb.Table(
                    columns=["input_sample", "generated_sample", "logreward"], data=rows
                )
            }
        )

        return loss

    def validation_step(self, val_batch, batch_idx) -> Any:
        x, state = self.sample(val_batch, train=False, epsilon=None, temperature=None)

        loss, logreward, logZ = self.compute_loss(x, state)
        self.val_loss(loss)
        self.val_logreward(logreward.mean())
        self.val_logZ(logZ.mean())

        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)
        self.log(
            "val/logreward",
            self.val_logreward,
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
