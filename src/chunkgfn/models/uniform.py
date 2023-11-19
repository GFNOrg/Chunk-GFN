import torch
from torch import nn
from torchtyping import TensorType as TT


class SeqDiscreteUniform(nn.Module):
    """Implements a uniform distribution over discrete actions.

    It uses a zero function approximator (a function that always outputs 0) to be used as
    logits by a DiscretePBEstimator.

    Attributes:
        output_dim: The size of the output space.
    """

    def __init__(self, output_dim: int) -> None:
        """Initializes the uniform function approximiator.

        Args:
            output_dim (int): Output dimension. This is typically n_actions if it
                implements a Uniform PF, or n_actions-1 if it implements a Uniform PB.
        """
        super().__init__()
        self.output_dim = output_dim

    def forward(
        self, preprocessed_states: TT["batch_shape", "input_dim", float]
    ) -> TT["batch_shape", "output_dim", float]:
        out = (
            torch.zeros(*preprocessed_states.shape[:-1], self.output_dim)
            .to(preprocessed_states.device)
            .mean(-2)
        )
        return out
