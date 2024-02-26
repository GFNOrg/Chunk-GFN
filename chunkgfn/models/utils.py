import torch
from torch import nn


def expand_linear_layer(
    layer: nn.Module, new_in_dim: int | None = None, new_out_dim: int | None = None
):
    old_out_dim, old_in_dim = layer.weight.shape
    if new_in_dim is not None:
        assert (
            new_in_dim > old_in_dim
        ), "new_in_dim must be bigger than the layer's in features"
        in_dim = new_in_dim
    else:
        in_dim = old_in_dim
    if new_out_dim is not None:
        assert (
            new_out_dim > old_out_dim
        ), "new_out_dim must be bigger than the layer's out features"
        out_dim = new_out_dim
    else:
        out_dim = old_out_dim

    new_layer = nn.Linear(in_dim, out_dim).to(layer.weight.device)
    with torch.no_grad():
        new_layer.weight[:old_out_dim, :old_in_dim] = layer.weight
        new_layer.bias[:old_out_dim] = layer.bias
    return new_layer
