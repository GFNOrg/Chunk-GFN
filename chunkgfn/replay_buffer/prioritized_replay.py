import torch
import torch.nn.functional as F
from einops import repeat

from . import ReplayBuffer


def distance(src: torch.Tensor, dst: torch.Tensor):
    """Compute the distance between two tensors.
    Args:
        src (torch.Tensor[...]): source tensor
        dst (torch.Tensor[n_samples, ...]): destination tensor
    """
    assert (
        src.shape == dst.shape[1:]
    ), "The source tensor must have the same shape as the destination tensor without the first dimension."
    src_ = repeat(src, "... -> n_samples ...", n_samples=dst.shape[0])
    src_ = src_.reshape(dst.shape[0], -1)
    dst_ = dst.reshape(dst.shape[0], -1)
    return F.mse_loss(src_, dst_, reduction="none").sum(-1)


class PrioritizedReplay(ReplayBuffer):
    def __init__(self, cutoff_distance: float, capacity: int = 1000):
        super().__init__(capacity)
        self.cutoff_distance = cutoff_distance

    def add(
        self,
        input: torch.Tensor,
        trajectories: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
        final_state: torch.Tensor,
        logreward: torch.Tensor,
    ):
        """Add samples to the replay buffer.
        Assumes all arguments to be torch tensors and that the replay buffer is sorted in ascending order.
        Args:
            input (torch.Tensor[n_samples, max_len, input_dim]): Input to the model.
            trajectories (torch.Tensor[n_samples, traj_len, max_len, state_dim]): Trajectories.
            actions (torch.Tensor[n_samples, traj_len, action_dim]): Actions.
            dones (torch.Tensor[n_samples, traj_len]): Whether trajectory is over.
            final_state (torch.Tensor[n_samples, max_len, state_dim]): Final state.
            logreward (torch.Tensor[n_samples]): Log reward.
        """
        assert all(
            isinstance(arg, torch.Tensor)
            for arg in [input, trajectories, actions, dones, final_state, logreward]
        ), "All elements must be torch tensors!"

        input = input.cpu()
        trajectories = trajectories.cpu()
        actions = actions.cpu()
        dones = dones.cpu()
        final_state = final_state.cpu()
        logreward = logreward.cpu()

        if len(self) == 0:
            self.storage = {
                "input": input,
                "trajectories": trajectories,
                "actions": actions,
                "dones": dones,
                "final_state": final_state,
                "logreward": logreward,
            }
            # Sort elements by logreward
            ix = torch.argsort(logreward)
            for key in self.storage.keys():
                self.storage[key] = self.storage[key][ix]
        elif len(self) < self.capacity:
            self.storage["input"] = torch.cat([self.storage["input"], input], dim=0)
            self.storage["trajectories"] = torch.cat(
                [self.storage["trajectories"], trajectories], dim=0
            )
            self.storage["actions"] = torch.cat(
                [self.storage["actions"], actions], dim=0
            )
            self.storage["dones"] = torch.cat([self.storage["dones"], dones], dim=0)
            self.storage["final_state"] = torch.cat(
                [self.storage["final_state"], final_state], dim=0
            )
            self.storage["logreward"] = torch.cat(
                [self.storage["logreward"], logreward], dim=0
            )
            # Sort elements by logreward
            ix = torch.argsort(self.storage["logreward"])
            for key in self.storage.keys():
                self.storage[key] = self.storage[key][ix]
                self.storage[key] = self.storage[key][-self.capacity :]
        else:
            dict_curr_batch = {
                "input": input,
                "trajectories": trajectories,
                "actions": actions,
                "dones": dones,
                "final_state": final_state,
                "logreward": logreward,
            }
            for i in range(len(final_state)):
                # This assumes that the replay buffer is ordered in ascending log-reward !
                if logreward[i] > self.storage["logreward"][0]:
                    src = torch.cat([input[i], final_state[i]], dim=-1)
                    dst = torch.cat(
                        [self.storage["input"], self.storage["final_state"]], dim=-1
                    )
                    delta = distance(src, dst)
                    if min(delta) > self.cutoff_distance:
                        self.storage = {
                            key: value[1:] for (key, value) in self.storage.items()
                        }
                        ix = torch.searchsorted(self.storage["logreward"], logreward[i])
                        for key in self.storage.keys():
                            self.storage[key] = torch.cat(
                                (
                                    self.storage[key][:ix],
                                    dict_curr_batch[key][i : i + 1],
                                    self.storage[key][ix:],
                                ),
                                axis=0,
                            )
                    else:
                        # Find nearest neighbor
                        ix_nn = torch.argmin(delta)
                        if self.storage["logreward"][ix_nn] < logreward[i]:
                            for key in self.storage.keys():
                                self.storage[key][ix_nn] = dict_curr_batch[key][i]
                        ix = torch.argsort(self.storage["logreward"])
                        for key in self.storage.keys():
                            self.storage[key] = self.storage[key][ix]

    def sample(self, num_samples: int):
        """Sample from the replay buffer according to the logreward and without replacement.
        Args:
            num_samples (int): Number of samples to draw.
        Returns:
            dict: Dictionary containing the samples.
        """
        probs = torch.softmax(self.storage["logreward"], dim=-1)
        ixs = torch.multinomial(probs, num_samples, replacement=False)
        samples = {}

        for key in self.storage.keys():
            samples[key] = self.storage[key][ixs]
        return samples
