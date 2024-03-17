import torch

from .base_replay_buffer import ReplayBuffer
from .utils import extend_trajectories


class RandomReplay(ReplayBuffer):
    def __init__(
        self,
        reward_sampling: bool = False,
        capacity: int = 1000,
        is_conditional: bool = True,
    ):
        super().__init__(capacity, is_conditional)
        self.reward_sampling = (
            reward_sampling  # Whether to sample according to the reward
        )

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
            actions (torch.Tensor[n_samples, traj_len]): Actions.
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
        else:
            new_trajetcories, new_actions, new_dones = extend_trajectories(
                self.storage["trajectories"],
                trajectories,
                self.storage["actions"],
                actions,
                self.storage["dones"],
                dones,
            )
            self.storage["input"] = torch.cat([self.storage["input"], input], dim=0)

            self.storage["actions"] = new_actions
            self.storage["trajectories"] = new_trajetcories
            self.storage["dones"] = new_dones
            self.storage["final_state"] = torch.cat(
                [self.storage["final_state"], final_state], dim=0
            )
            self.storage["logreward"] = torch.cat(
                [self.storage["logreward"], logreward], dim=0
            )

            for key in self.storage.keys():
                self.storage[key] = self.storage[key][-self.capacity :]

    def sample(self, num_samples: int):
        """Sample from the replay buffer randomly or according to the logreward
        and without replacement.
        Args:
            num_samples (int): Number of samples to draw.
        Returns:
            dict: Dictionary containing the samples.
        """
        if self.reward_sampling:
            probs = torch.softmax(self.storage["logreward"], dim=-1)
            indices = torch.multinomial(probs, num_samples, replacement=False)
        else:
            indices = torch.randperm(len(self))[:num_samples]
        samples = {}
        for key in self.storage.keys():
            samples[key] = self.storage[key][indices]

        return samples
