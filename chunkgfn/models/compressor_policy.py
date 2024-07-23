import torch

from .base_policy import BasePolicy


class CompressorPolicy(BasePolicy):
    def __init__(self, alpha: float = 0):
        super().__init__()
        self.alpha = alpha

    def forward(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        env = self._environment
        logN = env.compute_logN(state, self.alpha)
        logits = env.get_logpb_state(logN, state, self.alpha)
        return logits
