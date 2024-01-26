from abc import ABC, abstractmethod

import torch
from torch import Tensor


def get_ix_unique(x: torch.Tensor):
    """Get indices of unique elements in a tensor.
    Args:
        x (torch.Tensor[batch_size, ...]): Tensor.
    Returns:
        torch.Tensor[n_unique]: Indices of unique elements.
    """
    x = x.view(x.shape[0], -1)
    unique_elements = torch.unique(x, dim=0)
    unique_elements_ = unique_elements.unsqueeze(1)
    x_ = x.unsqueeze(0)
    M = (unique_elements_ == x_).all(dim=-1)
    return (M + 0).argmax(dim=1)


class ReplayBuffer(ABC):
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.storage = {
            "input": torch.Tensor(),
            "trajectories": torch.Tensor(),
            "actions": torch.Tensor(),
            "dones": torch.Tensor(),
            "final_state": torch.Tensor(),
            "logreward": torch.Tensor(),
        }

    def __len__(self):
        return len(self.storage["input"])

    def pad_dim(self, tensor: torch.Tensor, new_dim: int):
        """Fix the dimension of a tensor by padding with zeros.
        Args:
            tensor (torch.Tensor[..., old_dim]): Tensor to be padded.
            new_dim (int): New dimension.
        Returns:
            torch.Tensor: Padded tensor.
        """
        assert new_dim >= tensor.size(-1), "New dimension must be larger than old one!"
        new_tensor = torch.zeros(*tensor.size()[:-1], new_dim).to(tensor)
        new_tensor[..., : tensor.size(-1)] = tensor
        return new_tensor

    def add(
        self,
        input: torch.Tensor,
        trajectories: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
        final_state: torch.Tensor,
        logreward: torch.Tensor,
    ):
        """Add samples to the replay buffer. Assumes all arguments to be torch tensors.
        Args:
            input (torch.Tensor[batch_size, max_len, input_dim]): Input to the model.
            trajectories (torch.Tensor[batch_size, traj_len, max_len, state_dim]): Trajectories.
            actions (torch.Tensor[batch_size, traj_len, action_dim]): Actions.
            dones (torch.Tensor[batch_size, traj_len]): Whether trajectory is over.
            final_state (torch.Tensor[batch_size, max_len, state_dim]): Final state.
            logreward (torch.Tensor[batch_size]): Log reward.
        """
        assert all(
            isinstance(arg, Tensor)
            for arg in [input, trajectories, actions, dones, final_state, logreward]
        ), "All elements must be torch tensors!"

        input = input.cpu()
        trajectories = trajectories.cpu()
        actions = actions.cpu()
        dones = dones.cpu()
        final_state = final_state.cpu()
        logreward = logreward.cpu()
        # Pad all state/action related tensors to the same dimension
        for key in ["trajectories", "actions", "final_state"]:
            self.storage[key] = self.pad_dim(self.storage[key], actions.shape[-1])

        # Concatenate all tensors
        if len(self) == 0:
            self.storage["input"] = input
            self.storage["trajectories"] = trajectories
            self.storage["actions"] = actions
            self.storage["dones"] = dones
            self.storage["final_state"] = final_state
            self.storage["logreward"] = logreward
        else:
            self.storage["input"] = torch.cat([self.storage["input"], input])
            self.storage["trajectories"] = torch.cat(
                [self.storage["trajectories"], trajectories]
            )
            self.storage["actions"] = torch.cat([self.storage["actions"], actions])
            self.storage["dones"] = torch.cat([self.storage["dones"], dones])
            self.storage["final_state"] = torch.cat(
                [self.storage["final_state"], final_state]
            )
            self.storage["logreward"] = torch.cat(
                [self.storage["logreward"], logreward]
            )

        # Remove duplicates indicated by input and final_state jointly
        indices = get_ix_unique(
            torch.cat(
                [
                    self.storage["input"],
                    self.storage["final_state"],
                ],
                dim=-1,
            )
        )
        for key in self.storage.keys():
            self.storage[key] = self.storage[key][indices]

        # Keep only the a maximum of self.capacity samples
        self.keep_capacity()

    @abstractmethod
    def sample(self, num_samples: int):
        NotImplementedError

    @abstractmethod
    def keep_capacity(self):
        NotImplementedError
