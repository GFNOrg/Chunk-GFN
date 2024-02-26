from abc import ABC, abstractmethod

import torch


class ConditionalReplayBuffer(ABC):
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

    @abstractmethod
    def sample(self, num_samples: int):
        NotImplementedError

    @abstractmethod
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
        NotImplementedError
