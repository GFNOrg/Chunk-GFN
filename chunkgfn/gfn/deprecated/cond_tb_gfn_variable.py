from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.optim import lr_scheduler as lr_scheduler
from torch.optim.optimizer import Optimizer as Optimizer

from chunkgfn.gfn.base_conditional_gfn import ConditionalSequenceGFN
from chunkgfn.gfn.utils import pad_dim
from chunkgfn.models.utils import expand_embedding_layer, expand_linear_layer
from chunkgfn.replay_buffer import ReplayBuffer
from chunkgfn.schedulers import Scheduler


class Cond_TBGFN_Variable(ConditionalSequenceGFN):
    def __init__(
        self,
        forward_model: nn.Module,
        backward_model: nn.Module,
        partition_model: nn.Module,
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
            partition_model,
            optimizer,
            scheduler,
            epsilon_scheduler,
            temperature_scheduler,
            replay_buffer,
            **kwargs,
        )
        self.automatic_optimization = False

    def go_backward(
        self,
        x: torch.Tensor,
        final_state: torch.Tensor,
    ):
        """Sample backward trajectories conditioned on inputs.
        Args:
            x (torch.Tensor[batch_size, seq_length, input_dim]): Conditioning vector.
            final_state (torch.Tensor[batch_size, seq_length, input_dim]): Final state.
        Return:
            trajectories (torch.Tensor[batch_size, trajectory_length, seq_length, state_dim]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            state (torch.Tensor[batch_size, seq_length, state_dim]): Final state.
        """
        bs, max_len, dim = x.shape
        eos_token_idx = self.trainer.datamodule.data_train.eos_token_idx
        state = final_state.clone()
        eos_token = torch.zeros(dim).to(x)
        eos_token[eos_token_idx] = 1
        done = torch.ones((bs)).to(x).bool()

        # Start unrolling the trajectories
        actions = []
        trajectories = []
        dones = []
        dones.append(torch.ones((bs)).to(x).bool())

        for t in range(max_len):
            backward_actions = self.trainer.datamodule.get_parent_actions(state)
            logp_b_s = torch.where(
                backward_actions == 1,
                self.trainer.datamodule.data_train.action_len[backward_actions] - 1,
                -torch.inf,
            ).to(state)
            cat = Categorical(logits=logp_b_s)

            act = cat.sample()

            new_state, done = self.trainer.datamodule.backward_step(state, act)

            actions.append(act)
            trajectories.append(state)
            dones.append(done.clone())

            state = new_state.clone()

        trajectories.append(state)

        trajectories = torch.stack(trajectories[::-1], dim=1)
        actions = torch.stack(actions[::-1], dim=1)
        dones = torch.stack(dones[::-1], dim=1)

        return trajectories, actions, dones, final_state

    def compute_loss(self, x, trajectories, actions, dones, logreward):
        """Compute the loss for the model.
        Args:
            x (torch.Tensor[batch_size, seq_length, input_dim]): Conditioning vector.
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
            logp_f_s = self.forward_model(x, state)
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

        logZ = self.partition_model(x).squeeze(-1)
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

        self.forward_model.action_embeddings = expand_embedding_layer(
            self.forward_model.action_embeddings,
            new_in_dim=self.forward_model.action_embeddings.data.shape[0] + 1,
            init_weights=None,
        )
        """
        self.forward_model.logits_layer = expand_linear_layer(
            self.forward_model.logits_layer,
            new_out_dim=self.forward_model.logits_layer.out_features + 1,
            init_weights=init_weights,
        )
        """
        params_group = self.trainer.optimizers[0].state_dict()["param_groups"]
        state = self.trainer.optimizers[0].state_dict()["state"]
        state[0]["exp_avg"] = pad_dim(
            state[0]["exp_avg"].T, self.forward_model.action_embeddings.data.shape[0]
        ).T
        state[0]["exp_avg_sq"] = pad_dim(
            state[0]["exp_avg_sq"].T, self.forward_model.action_embeddings.data.shape[0]
        ).T
        # Reinitialize the optimizer
        self.trainer.optimizers = [self.configure_optimizers()["optimizer"]]
        self.trainer.optimizers[0].state_dict()["param_groups"] = params_group
        self.trainer.optimizers[0].state_dict()["state"] = state

    def training_step(self, train_batch, batch_idx) -> Any:
        loss = super().training_step(train_batch, batch_idx)

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
