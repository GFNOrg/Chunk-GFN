from abc import ABC, abstractmethod

import torch
from torch import Tensor

from chunkgfn.gfn.utils import pad_dim


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
            self.storage[key] = pad_dim(self.storage[key], actions.shape[-1])

        dict_curr_batch = { "input": input,
            "trajectories": trajectories,
            "actions": actions,
            "dones": dones,
            "final_state": final_state,
            "logreward": logreward}
        for i in range(len(final_state)):
            if dict_curr_batch['logreward'][i] > self.storage['logreward'][0]: # This assumes that the replay buffer is ordered in ascending log-reward ! 
                delta = dist(input[i], dict_curr_batch['final_state'][i], self.storage) # size : (len(replay_buffer))
                if min(delta) > self.thresh_dist : 
                    self.storage = {key: value[1:] for (key, value) in self.storage.items()}
                    ix = torch.searchsorted(self.storage('logreward'),dict_curr_batch['logreward'][i] )
                    for key in self.storage.keys():
                        self.storage[key] = torch.cat((self.storage[key][:ix] ,  dict_curr_batch[key][i:i+1], self.storage[key][ix:] ) , axis = 0 )
                else: 
                    #Find nearest neighbor
                    ix_nn = torch.argmin(delta)
                    if self.storage['logreward'][ix_nn] < logreward[i]:  
                        for key in self.storage.keys():
                            self.storage[key][ix_nn] = self.dict_curr_batch[key][i]


                    


        # Keep only the a maximum of self.capacity samples
        self.keep_capacity()

    @abstractmethod
    def sample(self, num_samples: int):
        NotImplementedError

    @abstractmethod
    def keep_capacity(self):
        NotImplementedError
