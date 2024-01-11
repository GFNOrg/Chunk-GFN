import torch

from . import ReplayBuffer


class RandomReplay(ReplayBuffer):
    def sample(self, num_samples: int):
        indices = torch.randperm(len(self))[:num_samples]
        samples = tuple([element[indices] for element in self.storage])

        return samples
