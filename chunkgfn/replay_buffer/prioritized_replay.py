import torch
from torch import Tensor

from . import ReplayBuffer


class PrioritizedReplay(ReplayBuffer):
    def add(self, *args):
        """Add samples to the replay buffer. Assumes all arguments to be torch tensors."""
        assert all(
            isinstance(arg, Tensor) for arg in args
        ), "All elements must be torch tensors!"
        if self.storage is None:
            self.storage = tuple(args)
        else:
            self.storage = tuple(
                [torch.cat([place, arg]) for arg, place in zip(args, self.storage)]
            )

            dist = torch.distributions.Categorical(logits=self.storage[-1] / 10)
            size = min(self.storage[-1].size(0), self.capacity)
            ixs = dist.sample((size,))
            self.storage = tuple([element[ixs] for element in self.storage])

    def sample(self, num_samples: int):
        dist = torch.distributions.Categorical(logits=self.storage[-1] / 10)
        ixs = dist.sample((num_samples,))
        samples = tuple([element[ixs] for element in self.storage])

        return samples
