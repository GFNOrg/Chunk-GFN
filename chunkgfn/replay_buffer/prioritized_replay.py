import torch

from .base_replay_buffer import ReplayBuffer
from .utils import extend_trajectories


def distance(src: torch.Tensor, dst: torch.Tensor):
    """Compute the squared distance between two tensors.
    Args:
        src (torch.Tensor[...]): source tensor
        dst (torch.Tensor[n_samples, ...]): destination tensor
    """
    assert (
        src.shape == dst.shape[1:]
    ), "The source tensor must have the same shape as the destination tensor without the first dimension."
    return ((src.unsqueeze(0) - dst) ** 2).sum(-1).sum(-1)


class PrioritizedReplay(ReplayBuffer):
    def __init__(
        self, cutoff_distance: float, capacity: int = 1000, is_conditional: bool = True
    ):
        super().__init__(capacity, is_conditional)
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

        # This is the first batch.
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

        # Adding a batch and the buffer isn't full yet.
        elif len(self) < self.capacity:
            new_trajectories, new_actions, new_dones = extend_trajectories(
                self.storage["trajectories"],
                trajectories,
                self.storage["actions"],
                actions,
                self.storage["dones"],
                dones,
            )
            self.storage["input"] = torch.cat([self.storage["input"], input], dim=0)
            self.storage["trajectories"] = new_trajectories
            self.storage["actions"] = new_actions
            self.storage["dones"] = new_dones
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
                # Ensures that the buffer is the correct size.
                self.storage[key] = self.storage[key][-self.capacity :]

        # Our buffer is full and we will prioritize diverse, high reward additions.
        else:
            dict_curr_batch = {
                "input": input,
                "trajectories": trajectories,
                "actions": actions,
                "dones": dones,
                "final_state": final_state,
                "logreward": logreward,
            }

            def _apply_idx(idx, d):
                for k, v in d.items():
                    d[k] = v[idx, ...]

            # Sort elements by logreward.
            idx_sorted = torch.argsort(dict_curr_batch["logreward"], descending=True)
            _apply_idx(idx_sorted, dict_curr_batch)

            # Filter all batch logrewards lower than the smallest logreward in buffer.
            idx_min_lr = dict_curr_batch["logreward"] > self.storage["logreward"].min()
            _apply_idx(idx_min_lr, dict_curr_batch)

            # Compute all pairwise distances between the batch and the buffer.
            curr_dim = dict_curr_batch["final_state"].shape[0]
            buffer_dim = self.storage["final_state"].shape[0]
            if curr_dim > 0:
                # Distances should incorporate conditioning vector.
                if self.is_conditional:
                    batch = torch.cat(
                        [dict_curr_batch["input"], dict_curr_batch["final_state"]],
                        dim=-1,
                    )
                    buffer = torch.cat(
                        [self.storage["input"], self.storage["final_state"]],
                        dim=-1,
                    )
                else:
                    batch = dict_curr_batch["final_state"].float()
                    buffer = self.storage["final_state"].float()

                # Filter the batch for diverse final_states with high reward.
                batch_batch_dist = torch.cdist(
                    batch.view(curr_dim, -1).unsqueeze(0),
                    batch.view(curr_dim, -1).unsqueeze(0),
                    p=1.0,
                ).squeeze(0)

                r, w = torch.triu_indices(*batch_batch_dist.shape)  # Remove upper diag.
                batch_batch_dist[r, w] = torch.finfo(batch_batch_dist.dtype).max
                batch_batch_dist = batch_batch_dist.min(-1)[0]

                # Filter the batch for diverse final_states w.r.t the buffer.
                batch_buffer_dist = (
                    torch.cdist(
                        batch.view(curr_dim, -1).unsqueeze(0),
                        buffer.view(buffer_dim, -1).unsqueeze(0),
                        p=1.0,
                    )
                    .squeeze(0)
                    .min(-1)[0]
                )

                # Remove non-diverse examples accordin to the above distances.
                idx_batch_batch = batch_batch_dist > self.cutoff_distance
                idx_batch_buffer = batch_buffer_dist > self.cutoff_distance
                idx_diverse = idx_batch_batch & idx_batch_buffer
                _apply_idx(idx_diverse, dict_curr_batch)

            # Concatenate everything, sort, and remove leftovers.
            for k, v in self.storage.items():
                if k not in ["trajectories", "actions", "dones"]:
                    self.storage[k] = torch.cat(
                        (self.storage[k], dict_curr_batch[k]), dim=0
                    )
            new_trajectories, new_actions, new_dones = extend_trajectories(
                self.storage["trajectories"],
                dict_curr_batch["trajectories"],
                self.storage["actions"],
                dict_curr_batch["actions"],
                self.storage["dones"],
                dict_curr_batch["dones"],
            )
            self.storage["trajectories"] = new_trajectories
            self.storage["actions"] = new_actions
            self.storage["dones"] = new_dones

            idx_sorted = torch.argsort(self.storage["logreward"], descending=False)
            _apply_idx(idx_sorted, self.storage)

            for k, v in self.storage.items():
                self.storage[k] = self.storage[k][-self.capacity :]  # Keep largest.

    def sample(self, num_samples: int):
        """Sample from the replay buffer according to the logreward and without replacement.
        Args:
            num_samples (int): Number of samples to draw.
        Returns:
            dict: Dictionary containing the samples.
        """
        probs = torch.softmax(self.storage["logreward"], dim=-1)
        if probs.shape[-1]< num_samples:
            raise ValueError(f"Number of samples to draw is larger than the buffer size. Decrease gfn.num_samples to less than {probs.shape[-1]}")
        ixs = torch.multinomial(probs, num_samples, replacement=False)
        samples = {}

        for key in self.storage.keys():
            samples[key] = self.storage[key][ixs]
        return samples
