import lightning as L
import pytest
import torch

from chunkgfn.datamodules.bit_sequence import BitSequenceModule


@pytest.mark.parametrize("num_modes", [60, 120, 240])
@pytest.mark.parametrize("oracle_difficulty", ["medium", "hard"])
@pytest.mark.parametrize("max_len", [32, 64, 128])
@pytest.mark.parametrize("seed", [42, 2024, 7, 1234])
def test_length_of_modes(
    num_modes: int, oracle_difficulty: str, max_len: int, seed: int
):
    L.seed_everything(seed, workers=True)
    env = BitSequenceModule(
        max_len=max_len,
        num_modes=num_modes,
        num_train_iterations=1000,
        num_val_iterations=100,
        num_test_iterations=100,
        threshold=0.9,
        oracle_difficulty=oracle_difficulty,
        batch_size=64,
        sample_exact_length=True,
    )
    assert (
        len(env.modes) == num_modes
    ), f"The number of modes is not correct. You asked for {num_modes} modes, but got {len(env.modes)} modes."


@pytest.mark.parametrize("max_len", [32, 64, 128])
@pytest.mark.parametrize("seed", [42, 2024, 7, 1234])
def test_invalid_actions_mask_exact_length(seed: int, max_len: int):
    L.seed_everything(seed, workers=True)
    env = BitSequenceModule(
        max_len=max_len,
        num_modes=60,
        num_train_iterations=1000,
        num_val_iterations=100,
        num_test_iterations=100,
        threshold=0.9,
        oracle_difficulty="medium",
        batch_size=64,
        sample_exact_length=True,
    )
    s0 = env.s0.unsqueeze(0)
    mask = env.get_invalid_actions_mask(s0)
    assert (
        tuple(mask.shape) == (s0.shape[0], len(env.actions))
    ), f"Invalid mask shape is not correct. Expected {(s0.shape[0], len(env.actions))} but got {tuple(mask.shape)}."
    expected_mask = torch.ones(len(env.actions))
    expected_mask[0] = 0
    expected_mask = expected_mask.unsqueeze(0)
    assert torch.all(
        mask == expected_mask
    ), "Invalid mask is not correct. The <EOS> should not be allowed."

    # Test for a state that is full
    idx = torch.randint(1, len(env.atomic_tokens), (64, max_len + 1))
    state = torch.zeros(64, max_len + 1, len(env.atomic_tokens)).scatter_(
        2, idx.unsqueeze(-1), 1
    )
    state[:, -1] = env.padding_token
    mask = env.get_invalid_actions_mask(state)
    expected_mask = torch.ones(len(env.actions))
    expected_mask[1:] = 0
    expected_mask = expected_mask.unsqueeze(0)
    assert torch.all(
        mask == expected_mask
    ), "Invalid mask is not correct. Only <EOS> token should be allowed."
