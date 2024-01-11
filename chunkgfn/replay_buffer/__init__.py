from abc import ABC, abstractmethod

import torch
from torch import Tensor


class ReplayBuffer(ABC):
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.storage = None

    def __len__(self):
        if self.storage is None:
            return 0
        return len(self.storage[0])

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
            self.storage = self.storage[-self.capacity :]

    @abstractmethod
    def sample(self, num_samples: int):
        NotImplementedError
