import torch

from . import ReplayBuffer


class PrioritizedReplay(ReplayBuffer):
    def keep_capacity(self):
        dist = torch.distributions.Categorical(logits=self.storage["logreward"])
        if len(self) > self.capacity:
            ixs = dist.sample((self.capacity,))
            for key in self.storage.keys():
                self.storage[key] = self.storage[key][ixs]

    def sample(self, num_samples: int):
        dist = torch.distributions.Categorical(logits=self.storage["logreward"])
        ixs = dist.sample((num_samples,))
        samples = {}
        for key in self.storage.keys():
            samples[key] = self.storage[key][ixs]
        return samples
