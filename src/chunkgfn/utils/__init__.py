import logging
import random
import warnings
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random. Adapted
    from https://pytorch-lightning.readthedocs.io/en/1.7.7/_modules/pytorch_lightning/utilities/seed.html#seed_everything

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will select seed randomly.

    """
    if seed is None:
        seed = random.randint(min_seed_value, max_seed_value)
        warnings.warn(f"No seed found, seed set to {seed}", stacklevel=4)

    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        warnings.warn(
            f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}",
            stacklevel=4,
        )
        seed = random.randint(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    log.info(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed
