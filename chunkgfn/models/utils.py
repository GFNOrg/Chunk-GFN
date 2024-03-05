from typing import Callable

import torch
from torch import nn


def expand_linear_layer(
    layer: nn.Module,
    new_in_dim: int | None = None,
    new_out_dim: int | None = None,
    init_weights: Callable | None = None,
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
        if init_weights is not None:
            init_weights(new_layer)

    return new_layer


def expand_embedding_layer(
    layer: nn.Module,
    new_in_dim: int | None = None,
    new_out_dim: int | None = None,
    init_weights: Callable | None = None,
):
    old_in_dim, old_out_dim = layer.data.shape
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
    new_layer = nn.Parameter(torch.randn(in_dim, out_dim).to(layer.data.device))
    with torch.no_grad():
        new_layer.data[:old_in_dim, :old_out_dim] = layer.data
        if init_weights is not None:
            init_weights(new_layer)
    return new_layer
