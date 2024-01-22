import torch

from . import ReplayBuffer


class RandomReplay(ReplayBuffer):
    def keep_capacity(self):
        size = min(len(self), self.capacity)
        for key in self.storage.keys():
            self.storage[key] = self.storage[key][-size:]

    def sample(self, num_samples: int):
        indices = torch.randperm(len(self))[:num_samples]
        samples = {}
        for key in self.storage.keys():
            samples[key] = self.storage[key][indices]

        return samples
