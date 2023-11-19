import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
from gfn.containers import ReplayBuffer
from gfn.env import Env
from gfn.gflownet import GFlowNet
from gfn.modules import GFNModule
from gfn.samplers import Sampler
from omegaconf import DictConfig

from src.chunkgfn.trainer import Trainer
from src.chunkgfn.utils import seed_everything
import wandb

log = logging.getLogger(__name__)


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the GFlownet.
    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed)

    log.info("Instantiating the environment...")
    env: Env = hydra.utils.instantiate(cfg.env)

    log.info("Instantiating the replay buffer...")
    replay_buffer: ReplayBuffer = hydra.utils.instantiate(cfg.replay_buffer, env=env)

    log.info("Instantiating the forward and backward estimators...")
    pf_estimator: GFNModule = hydra.utils.instantiate(
        cfg.forward,
        n_actions=env.n_actions,
        is_backward=False,
        preprocessor=env.preprocessor,
    )
    pb_estimator: GFNModule = hydra.utils.instantiate(
        cfg.backward,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=env.preprocessor,
    )

    log.info("Instantiating the GFlowNet...")
    gflownet: GFlowNet = hydra.utils.instantiate(
        cfg.gflownet, pf=pf_estimator, pb=pb_estimator
    )

    log.info("Instantiating the sampler...")
    sampler: Sampler = hydra.utils.instantiate(cfg.sampler, estimator=pf_estimator)

    log.info("Instantiating the trainer...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        env=env,
        pf_estimator=pf_estimator,
        pb_estimator=pb_estimator,
        sampler=sampler,
        gflownet=gflownet,
        replay_buffer=replay_buffer,
    )

    log.info("Starting training!")
    trainer.fit()

    metric_dict = trainer.metrics

    return metric_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    metric_dict = train(cfg)
    metric_value = metric_dict[cfg.get("optimized_metric")]
    return metric_value


if __name__ == "__main__":
    main()
