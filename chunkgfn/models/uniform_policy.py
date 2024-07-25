import torch

from .base_policy import BasePolicy


class UniformPolicy(BasePolicy):
    def __init__(self, action_embedding_dim: int):
        super().__init__()
        self.action_embedding_dim = action_embedding_dim

    def forward(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.zeros(
            state.shape[0], self.action_embedding_dim, device=state.device
        )
