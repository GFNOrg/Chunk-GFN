import torch

from .base_policy import BasePolicy


class UniformPolicy(BasePolicy):
    def __init__(self, action_embedding_dim):
        super().__init__()

    def forward(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        env = self._environment
        return torch.zeros(state.shape[0], env.n_actions, device=state.device)
