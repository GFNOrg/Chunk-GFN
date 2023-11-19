"""
AdaptediImplementations of the [Trajectory Balance loss](https://arxiv.org/abs/2201.13259) for conditional GFlowNets
"""

import torch
import torch.nn as nn
from torchtyping import TensorType as TT

from gfn.containers import Trajectories
from gfn.env import Env
from gfn.gflownet.base import TrajectoryBasedGFlowNet
from gfn.modules import GFNModule


class ConditionalTBGFlowNet(TrajectoryBasedGFlowNet):
    r"""Holds the logZ estimate for the Trajectory Balance loss.

    $\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the
    DAG. $\mathcal{O}_3$ is the set of backward probability functions consistent with
    the DAG, or a singleton thereof, if self.logit_PB is a fixed DiscretePBEstimator.

    Attributes:
        logZ: a LogZEstimator instance.
        log_reward_clip_min: minimal value to clamp the reward to.

    """

    def __init__(
        self,
        pf: GFNModule,
        pb: GFNModule,
        logZ: nn.Module,
        on_policy: bool = False,
        log_reward_clip_min: float = -12,  # roughly log(1e-5)
    ):
        super().__init__(pf, pb, on_policy=on_policy)
        self.logZ = logZ
        self.log_reward_clip_min = log_reward_clip_min
    
    def get_conditional_from_trajectories(self, trajectories: Trajectories) -> torch.Tensor:
        """Extract the conditioning vector from states.
        """
        batch_ndim = len(trajectories.states.batch_shape)
        x, _ = torch.chunk(trajectories.states.tensor, 2, batch_ndim)
        x = x[0].float() # Extract only the first step
        return x

    def loss(self, env: Env, trajectories: Trajectories) -> TT[0, float]:
        """Trajectory balance loss.

        The trajectory balance loss is described in 2.3 of
        [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259))

        Raises:
            ValueError: if the loss is NaN.
        """
        del env  # unused
        _, _, scores = self.get_trajectories_scores(trajectories)
        x = self.get_conditional_from_trajectories(trajectories)
        logZ = self.logZ(x)
        loss = (scores + logZ).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss