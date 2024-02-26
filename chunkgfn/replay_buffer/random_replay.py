import torch

from .base_replay_buffer import ConditionalReplayBuffer


class RandomReplay(ConditionalReplayBuffer):
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
        else:
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

            for key in self.storage.keys():
                self.storage[key] = self.storage[key][-self.capacity :]

    def sample(self, num_samples: int):
        indices = torch.randperm(len(self))[:num_samples]
        samples = {}
        for key in self.storage.keys():
            samples[key] = self.storage[key][indices]

        return samples
