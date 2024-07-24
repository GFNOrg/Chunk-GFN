import torch


def pad_dim(tensor: torch.Tensor, new_dim: int):
    """Fix the dimension of a tensor by padding with zeros.
    Args:
        tensor (torch.Tensor[..., old_dim]): Tensor to be padded.
        new_dim (int): New dimension.
    Returns:
        torch.Tensor: Padded tensor.
    """
    assert new_dim >= tensor.shape[-1], "New dimension must be larger than old one!"
    new_tensor = torch.zeros(*tensor.shape[:-1], new_dim).to(tensor)
    new_tensor[..., : tensor.shape[-1]] = tensor
    return new_tensor


def has_trainable_parameters(module: torch.nn.Module) -> bool:
    """Check if a module has trainable parameters.
    Args:
        module (torch.nn.Module): Module to be checked.
    Returns:
        bool: True if the module has trainable parameters, False otherwise.
    """
    return any(param.requires_grad for param in module.parameters())
