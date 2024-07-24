import lightning as L
import pytest
import torch

from chunkgfn.environment.hypergrid import HyperGridModule


@pytest.mark.parametrize("ndim", [2, 4, 6, 8])
@pytest.mark.parametrize("side_length", [5, 10, 15, 20])
@pytest.mark.parametrize("seed", [42, 2024, 7, 1234])
def test_invalid_actions_mask_final(seed: int, ndim: int, side_length: int):
    """Test that the invalid actions mask is correct for the final state.
    When we reach the final state, only the <EXIT> action should be allowed.
    """
    L.seed_everything(seed, workers=True)
    env = HyperGridModule(
        ndim=ndim,
        side_length=side_length,
        num_modes=int(((side_length - 1) / 4) ** ndim),
        R0=0.1,
        R1=0.5,
        R2=2,
        num_train_iterations=1000,
        batch_size=64,
    )
    state = torch.ones(ndim + 1) * (side_length - 1)
    state[-1] = 0
    state = state.unsqueeze(0)
    mask = env.get_forward_mask(state)
    assert (
        tuple(mask.shape) == (state.shape[0], len(env.actions))
    ), f"Invalid mask shape is not correct. Expected {(state.shape[0], len(env.actions))} but got {tuple(mask.shape)}."
    expected_mask = torch.zeros(len(env.actions))
    expected_mask[env.actions.index("<EXIT>")] = 1
    expected_mask = expected_mask.unsqueeze(0)
    assert torch.all(
        mask == expected_mask
    ), "Invalid mask is not correct. Only <EXIT> should be allowed."


@pytest.mark.parametrize("seed", [42, 2024, 7, 1234])
def test_hypergrid_invalid_actions(seed: int):
    L.seed_everything(seed, workers=True)
    env = HyperGridModule(
        ndim=2,
        side_length=3,
        num_modes=4,
        R0=0.1,
        R1=0.5,
        R2=2,
        num_train_iterations=1000,
        batch_size=64,
    )

    states = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 2, 0],
            [0, 2, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [1, 2, 0],
            [1, 2, 1],
            [2, 0, 0],
            [2, 0, 1],
            [2, 1, 0],
            [2, 1, 1],
            [2, 2, 0],
            [2, 2, 1],
        ]
    )
    expected_mask = torch.tensor(
        [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )

    mask = env.get_forward_mask(states)
    assert torch.all(mask == expected_mask), "Invalid mask is not correct."


@pytest.mark.parametrize("seed", [42, 2024, 7, 1234])
def test_hypergrid_parent_actions(seed: int):
    L.seed_everything(seed, workers=True)
    env = HyperGridModule(
        ndim=2,
        side_length=3,
        num_modes=4,
        R0=0.1,
        R1=0.5,
        R2=2,
        num_train_iterations=1000,
        batch_size=64,
    )

    states = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 2, 0],
            [0, 2, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [1, 2, 0],
            [1, 2, 1],
            [2, 0, 0],
            [2, 0, 1],
            [2, 1, 0],
            [2, 1, 1],
            [2, 2, 0],
            [2, 2, 1],
        ]
    )
    expected_mask = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
        ]
    )

    mask = env.get_parent_actions(states)
    assert torch.all(mask == expected_mask), "Parent mask is not correct."
