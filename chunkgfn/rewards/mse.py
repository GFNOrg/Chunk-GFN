from typing import Any

import torch.nn.functional as F


class MSE:
    def __call__(self, input, target) -> Any:
        logreward = -F.mse_loss(input, target, reduction="none")
        logreward = logreward.reshape(logreward.shape[0], -1).sum(-1)
        return logreward
