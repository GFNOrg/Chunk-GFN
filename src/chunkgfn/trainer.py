import torch
from addict import Dict
from gfn.containers import ReplayBuffer
from gfn.env import Env
from gfn.gflownet.base import GFlowNet
from gfn.modules import GFNModule
from gfn.samplers import Sampler
from torchmetrics import MeanMetric
from tqdm.rich import tqdm


class Trainer:
    def __init__(
        self,
        env: Env,
        pf_estimator: GFNModule,
        pb_estimator: GFNModule,
        sampler: Sampler,
        gflownet: GFlowNet,
        replay_buffer: ReplayBuffer,
        **kwargs,
    ):
        self.hparams = Dict(kwargs)
        self.env = env
        self.pf_estimator = pf_estimator.to(self.hparams.device)
        self.pb_estimator = pb_estimator.to(self.hparams.device)
        self.sampler = sampler
        self.gflownet = gflownet
        self.replay_buffer = replay_buffer
        self.optimizer = self.configure_optimizers()

        # Metrics stuff
        self.loss = MeanMetric().to(self.hparams.device)
        self.metrics = {}

    def configure_optimizers(self):
        params = [
            {
                "params": [
                    v
                    for k, v in dict(self.gflownet.named_parameters()).items()
                    if k != "logZ"
                ],
                "lr": self.hparams.lr,
            }
        ]
        if "logZ" in dict(self.gflownet.named_parameters()):
            params.append(
                {
                    "params": [dict(self.gflownet.named_parameters())["logZ"]],
                    "lr": self.hparams.lr_Z,
                }
            )
        optimizer = self.hparams.object.optimizer(params=params)

        return optimizer

    def training_pass(self):
        trajectories = self.sampler.sample_trajectories(
            self.env,
            states=self.env.States.from_batch_shape(
                (self.hparams.batch_size,), random=True
            ),
        )
        training_samples = self.gflownet.to_training_samples(trajectories)
        if self.replay_buffer is not None:
            with torch.no_grad():
                self.replay_buffer.add(training_samples)
                training_objects = self.replay_buffer.sample(
                    n_trajectories=self.hparams.batch_size
                )
        else:
            training_objects = training_samples

        self.optimizer.zero_grad()
        loss = self.gflownet.loss(self.env, training_objects)
        loss.backward()
        self.optimizer.step()

        return loss, training_objects, trajectories

    def fit(self):
        for iteration in tqdm(range(self.hparams.num_iterations)):
            gfn_loss, training_objects, trajectories = self.training_pass()
            self.metrics["loss"] = self.loss(gfn_loss).item()
            self.metrics["logZ"] = self.gflownet.logZ.item()
            self.metrics["logRewards"] = trajectories.log_rewards.mean().item()
            tqdm.write(f"{iteration}: {self.metrics}")
