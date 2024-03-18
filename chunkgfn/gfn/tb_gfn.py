import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from chunkgfn.gfn.base_conditional_gfn import ConditionalSequenceGFN
from chunkgfn.gfn.base_unconditional_gfn import UnConditionalSequenceGFN


class TBGFN(UnConditionalSequenceGFN):
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
            logp_f_s = self.forward_model(
                self.trainer.datamodule.preprocess_states(state)
            )
            if t < trajectories.shape[1] - 1:
                log_pf += (Categorical(logits=logp_f_s).log_prob(actions[:, t])) * (
                    ~dones[:, t] + 0
                )
            if t > 0:
                backward_actions = self.trainer.datamodule.get_parent_actions(state)
                logp_b_s = self.backward_model(
                self.trainer.datamodule.preprocess_states(state)
            )
                logp_b_s = torch.where(
                    backward_actions == 1, logp_b_s, -torch.inf
                ).to(logp_f_s)
                # When no action is available, just fill with uniform because it won't be picked anyway in the backward_step. Doing this avoids having nan when computing probabilities
                logp_b_s = torch.where(
                    (logp_b_s == -torch.inf).all(dim=-1).unsqueeze(1),
                    torch.tensor(0.0),
                    logp_b_s,
                )
                log_pb += torch.where(
                    dones[:, t] | self.trainer.datamodule.is_initial_state(state),
                    torch.tensor(0.0).to(logp_b_s.device),
                    Categorical(logits=logp_b_s).log_prob(actions[:, t - 1]),
                )

        logZ = self.logZ
        loss = F.mse_loss(
            logZ + log_pf, (logreward / self.hparams.reward_temperature) + log_pb
        )

        return loss, logZ


class Cond_TBGFN(ConditionalSequenceGFN):
    def compute_loss(self, x, trajectories, actions, dones, logreward):
        """Compute the loss for the model.
        Args:
            x (torch.Tensor[batch_size, *input_shape]): Conditioning vector.
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
            logp_f_s = self.forward_model(x, state)
            if t < trajectories.shape[1] - 1:
                log_pf += (Categorical(logits=logp_f_s).log_prob(actions[:, t])) * (
                    ~dones[:, t] + 0
                )
            if t > 0:
                backward_actions = self.trainer.datamodule.get_parent_actions(state)
                logp_b_s = self.backward_model(self.trainer.datamodule.preprocess_states(state))
                logp_b_s = torch.where(
                    backward_actions == 1, logp_b_s, -torch.inf
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
