import torch


def pad_dim(tensor: torch.Tensor, new_dim: int):
    """Fix the dimension of a tensor by padding with zeros.
    Args:
        tensor (torch.Tensor[..., old_dim]): Tensor to be padded.
        new_dim (int): New dimension.
    Returns:
        torch.Tensor: Padded tensor.
    """
    assert new_dim >= tensor.size(-1), "New dimension must be larger than old one!"
    new_tensor = torch.zeros(*tensor.size()[:-1], new_dim).to(tensor)
    new_tensor[..., : tensor.size(-1)] = tensor
    return new_tensor
