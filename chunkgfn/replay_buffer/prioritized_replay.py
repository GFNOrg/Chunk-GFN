import torch

from . import ReplayBuffer


class PrioritizedReplay(ReplayBuffer):
    def keep_capacity(self):
        probs = torch.softmax(self.storage["logreward"], dim=-1)
        if len(self) > self.capacity:
            ixs = torch.multinomial(probs, self.capacity, replacement=False)
            for key in self.storage.keys():
                self.storage[key] = self.storage[key][ixs]

    def sample(self, num_samples: int):
        dist = torch.distributions.Categorical(logits=self.storage["logreward"])
        ixs = dist.sample((num_samples,))
        samples = {}
        for key in self.storage.keys():
            samples[key] = self.storage[key][ixs]
        return samples
