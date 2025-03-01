from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from einops import repeat
from lightning import LightningModule
from torch.distributions import Categorical
from torchmetrics import MeanMetric

from chunkgfn.algo.utils import has_trainable_parameters
from chunkgfn.environment.base_module import BaseUnConditionalEnvironmentModule
from chunkgfn.replay_buffer.base_replay_buffer import ReplayBuffer
from chunkgfn.replay_buffer.utils import extend_trajectories
from chunkgfn.schedulers import Scheduler

from ..constants import EPS, NEGATIVE_INFINITY


class BaseSampler(ABC, LightningModule):
    """Abstract class for samplers. Its requires a forward policy
    as well as action_embedder. The role of this class is to implement basic
    methods and attributes for building the simplest sampler there is and an example
    is that of a random sampler. Classes that inherit from this one can add more modules
    and losses.
    """

    def __init__(
        self,
        forward_policy: torch.nn.Module,
        action_embedder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        epsilon_scheduler: Scheduler | None = None,
        temperature_scheduler: Scheduler | None = None,
        replay_buffer: ReplayBuffer | None = None,
        **kwargs,
    ):
        super().__init__()

        # Forward policy model
        self.forward_policy = forward_policy
        self.action_embedder = action_embedder

        # Off-policy training
        self.epsilon_scheduler = epsilon_scheduler
        self.temperature_scheduler = temperature_scheduler
        self.replay_buffer = replay_buffer

        # Metric managers
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_logreward = MeanMetric()
        self.val_logreward = MeanMetric()
        self.train_trajectory_length = MeanMetric()

    def setup(self, stage):
        self.env: BaseUnConditionalEnvironmentModule = self.trainer.datamodule
        # Attach environment to policies
        if self.forward_policy is not None:
            self.forward_policy.set_environment(self.env)

    def configure_optimizers(self):
        params = []
        if self.forward_policy is not None and has_trainable_parameters(
            self.forward_policy
        ):
            params.append(
                {
                    "params": self.forward_policy.parameters(),
                    "lr": self.hparams.forward_policy_lr,
                }
            )
        if self.action_embedder is not None and has_trainable_parameters(
            self.action_embedder
        ):
            params.append(
                {
                    "params": self.action_embedder.parameters(),
                    "lr": self.hparams.action_embedder_lr,
                }
            )

        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @torch.no_grad()
    def refactor_replay_buffer(self):
        """Refactor the replay buffer. By default, this method simply clears the replay
        buffer. Subclasses can override this method to implement more complex logic.
        """
        if self.replay_buffer is not None:
            self.replay_buffer.clear()

    def chunked_forward(self, n_samples):
        """Performs a chunked forward pass.
        Args:
            n_samples (int): Total number of samples to process.
            max_chunk_size (int): Maximum chunk size to process at once.
        Returns:
            actions (torch.Tensor[n_samples, ...]): Actions for each sample.
            dones (torch.Tensor[n_samples, ...]): Dones for each sample.
            (other outputs): Other outputs from the forward pass.
        """
        max_chunk_size = self.hparams.max_chunk_size
        num_chunks = (
            n_samples + max_chunk_size - 1
        ) // max_chunk_size  # Calculate number of chunks

        trajectories = None
        actions = None
        dones = None
        all_states = []
        all_trajectory_lengths = []

        for i in range(num_chunks):
            start_idx = i * max_chunk_size
            end_idx = min(start_idx + max_chunk_size, n_samples)

            # Perform the forward pass on the chunk
            (
                trajectories_chunk,
                actions_chunk,
                dones_chunk,
                state_chunk,
                trajectory_length_chunk,
            ) = self.forward(end_idx - start_idx)

            if i == 0:
                trajectories = trajectories_chunk.clone()
                actions = actions_chunk.clone()
                dones = dones_chunk.clone()
            else:
                trajectories, actions, dones = extend_trajectories(
                    trajectories,
                    trajectories_chunk,
                    actions,
                    actions_chunk,
                    dones,
                    dones_chunk,
                )

            all_states.append(state_chunk)
            all_trajectory_lengths.append(trajectory_length_chunk)

        # Concatenate all chunks to form the full outputs
        states = torch.cat(all_states, dim=0)
        trajectory_lengths = torch.cat(all_trajectory_lengths, dim=0)

        return trajectories, actions, dones, states, trajectory_lengths

    @torch.no_grad()
    def update_library(self):
        """Update the library. This function will do the following, in the following order:
        1. Pick a number of generated samples from the replay buffer.
        2. Concatenate replay buffer samples with on-policy samples.
        3. Apply a tokenizing algorithm on the action sequences
        4. Add new token(s) to the actions list.
        """

        # Pick a number of generated samples from the replay buffer
        if self.replay_buffer is not None:
            nsamples_replay = int(
                self.hparams.n_samples * self.hparams.ratio_from_replay_buffer
            )
            trajectories, actions, dones, _, _ = self.forward(
                self.hparams.n_samples - nsamples_replay
            )

            samples = self.replay_buffer.sample(nsamples_replay)
            trajectories_rb = samples["trajectories"]
            actions_rb = samples["actions"]
            dones_rb = samples["dones"]

            # Concatenate samples from the replay buffer and the on-policy samples
            _, actions, dones = extend_trajectories(
                trajectories.to(trajectories_rb),
                trajectories_rb,
                actions.to(actions_rb),
                actions_rb,
                dones.to(dones_rb),
                dones_rb,
            )
        else:
            _, actions, dones, _, _ = self.chunked_forward(self.hparams.n_samples)

        # Get the most valuable tokens.
        if self.hparams.chunk_algorithm == "bpe":
            if self.hparams.chunk_type == "basic":
                self.trainer.datamodule.chunk_bpe(
                    actions,
                    dones,
                    n_tokens_to_add=self.hparams.n_chunks,
                )
            elif self.hparams.chunk_type == "replacement":
                n = self.hparams.total_library_size - len(
                    self.trainer.datamodule.atomic_tokens
                )
                self.trainer.datamodule.chunk_bpe(
                    actions,
                    dones,
                    n_tokens_to_add=n,
                    remove_old=True,
                )
            else:
                raise Exception("chunk_type not in ['basic', 'replacement']")
        elif self.hparams.chunk_algorithm == "wordpiece":
            if self.hparams.chunk_type == "basic":
                self.trainer.datamodule.chunk_wordpiece(
                    actions,
                    dones,
                    n_tokens_to_add=self.hparams.n_chunks,
                )
            elif self.hparams.chunk_type == "replacement":
                n = self.hparams.total_library_size - len(
                    self.trainer.datamodule.atomic_tokens
                )
                self.trainer.datamodule.chunk_wordpiece(
                    actions,
                    dones,
                    n_tokens_to_add=n,
                    remove_old=True,
                )
            else:
                raise Exception("chunk_type not in ['basic', 'replacement']")
        elif self.hparams.chunk_algorithm == "uniform":
            if self.hparams.chunk_type == "basic":
                self.trainer.datamodule.chunk_uniform(
                    n_tokens_to_add=self.hparams.n_chunks
                )
            elif self.hparams.chunk_type == "replacement":
                n = self.hparams.total_library_size - len(
                    self.trainer.datamodule.atomic_tokens
                )
                self.trainer.datamodule.chunk_uniform(
                    n_tokens_to_add=n, remove_old=True
                )
            else:
                raise Exception("chunk_type not in ['basic', 'replacement']")
        else:
            raise Exception("chunk_algorithm not in ['bpe', 'wordpiece', 'uniform']")

    def get_library_embeddings(self):
        """Produce embedding for all actions in the library.
        Returns:
            library_embeddings (torch.Tensor[n_actions, action_embedding]): Embeddings for all actions.
        """
        action_indices = self.env.action_indices
        library_embeddings = []
        for action, indices in action_indices.items():
            library_embeddings.append(
                self.action_embedder(
                    torch.LongTensor(indices).to(self.device).unsqueeze(0)
                )
            )
        library_embeddings = torch.cat(library_embeddings, dim=0)
        return library_embeddings

    def get_forward_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get the forward logits for the given state.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): State.
        Return:
            logits (torch.Tensor[batch_size, n_actions]): Forward logits.
        """
        processed = self.env.preprocess_states(state)
        if isinstance(processed, tuple):
            action_embedding = self.forward_policy(*processed)
        else:
            action_embedding = self.forward_policy(processed)
        dim = action_embedding.shape[-1]
        library_embeddings = self.get_library_embeddings()
        logits = torch.einsum("bd, nd -> bn", action_embedding, library_embeddings) / (
            dim**0.5
        )  # Same as in softmax
        return logits

    def forward(
        self,
        batch_size: int,
        train: bool = True,
        epsilon: float | None = None,
        temperature: float | None = None,
    ):
        """Sample forward trajectories conditioned on inputs.
        Args:
            batch_size (int): Number of samples to generate.
            train (bool): Whether it's during train or eval. This makes sure that we don't sample off-policy during inference.
            epsilon (float|None): Epsilon value for epsilon greedy.
            temperature (float|None): Temperature value for tempering.
        Return:
            trajectories (torch.Tensor[batch_size, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            state (torch.Tensor[batch_size, *state_shape]): Final state.
            trajectory_length (torch.Tensor[batch_size]): Length of the trajectory for each sample in the batch.
        """
        s0 = self.env.s0.to(self.device)
        state = repeat(s0, " ... -> b ...", b=batch_size)
        bs = state.shape[0]

        # Start unrolling the trajectories
        actions = []
        trajectories = []
        dones = []
        done = torch.zeros((bs), dtype=torch.bool, device=self.device)
        trajectory_length = torch.zeros(
            (bs), dtype=torch.long, device=self.device
        )  # This tracks the length of trajectory for each sample in the batch

        while not done.all():
            logit_pf = self.get_forward_logits(state)
            uniform_dist_probs = torch.ones_like(logit_pf, device=self.device)

            forward_mask = self.env.get_forward_mask(state)

            logit_pf = torch.where(
                forward_mask,
                logit_pf,
                torch.tensor(NEGATIVE_INFINITY, device=self.device),
            )
            uniform_dist_probs = torch.where(
                forward_mask,
                uniform_dist_probs,
                torch.tensor(0.0, device=self.device),
            )

            if train:
                if temperature is not None:
                    logits = logit_pf / (EPS + temperature)
                else:
                    logits = logit_pf
                if epsilon is not None:
                    probs = torch.softmax(logits, dim=-1)
                    uniform_dist_probs = uniform_dist_probs / uniform_dist_probs.sum(
                        dim=-1, keepdim=True
                    )
                    probs = (1 - epsilon) * probs + epsilon * uniform_dist_probs
                    cat = Categorical(probs=probs)
                else:
                    cat = Categorical(logits=logits)
            else:
                cat = Categorical(logits=logit_pf)

            act = cat.sample()

            new_state, done = self.env.forward_step(state, act)
            trajectory_length += ~done  # Increment the length of the trajectory for each sample in the batch as long it's not done.

            actions.append(act)
            trajectories.append(state)
            dones.append(done.clone())

            state = new_state.clone()

        trajectories.append(state)
        dones.append(torch.ones((bs), dtype=torch.bool, device=self.device))
        trajectories = torch.stack(trajectories, dim=1)
        actions = torch.stack(actions, dim=1)
        dones = torch.stack(dones, dim=1)

        return trajectories, actions, dones, state, trajectory_length

    @abstractmethod
    def compute_loss(self, trajectories, actions, dones, logreward):
        """Compute the loss for the model.
        Args:
            trajectories (torch.Tensor[batch_size, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            logreward (torch.Tensor[batch_size]): Log reward.
        """

    def sample(
        self,
        batch: torch.Tensor,
        train: bool = True,
        epsilon: float = 0.0,
        temperature: float = 0.0,
        calculate_logreward: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch

        trajectories, actions, dones, final_state, trajectory_length = self.forward(
            x.shape[0], train=train, epsilon=epsilon, temperature=temperature
        )
        if calculate_logreward:
            logreward = self.env.compute_logreward(final_state).to(final_state.device)
        else:
            logreward = None
        return (
            x,
            trajectories,
            actions,
            dones,
            final_state,
            logreward,
            trajectory_length,
        )

    def training_step(self, train_batch, batch_idx) -> Any:
        if self.epsilon_scheduler is not None:
            epsilon = self.epsilon_scheduler.step(self.current_epoch)
        else:
            epsilon = None
        if self.temperature_scheduler is not None:
            temperature = self.temperature_scheduler.step(self.current_epoch)
        else:
            temperature = None

        x, trajectories, actions, dones, final_state, logreward, trajectory_length = (
            self.sample(
                train_batch,
                train=True,
                epsilon=epsilon,
                temperature=temperature,
            )
        )
        batch_size = x.shape[0]

        sampler_logreward = logreward

        if self.replay_buffer is not None:
            nsamples_replay = int(batch_size * self.hparams.ratio_from_replay_buffer)
            with torch.no_grad():
                self.replay_buffer.add(
                    input=x,
                    trajectories=trajectories,
                    actions=actions,
                    dones=dones,
                    final_state=final_state,
                    logreward=logreward,
                )
                samples = self.replay_buffer.sample(nsamples_replay)

            for key in samples.keys():
                samples[key] = samples[key].to(x.device)

            # Concatenate samples from the replay buffer and the on-policy samples
            indices = torch.randperm(len(x))[: batch_size - nsamples_replay]
            x = torch.cat([x[indices], samples["input"]], dim=0)
            trajectories, actions, dones = extend_trajectories(
                trajectories[indices],
                samples["trajectories"],
                actions[indices],
                samples["actions"],
                dones[indices],
                samples["dones"],
            )

            final_state = torch.cat(
                [final_state[indices], samples["final_state"]], dim=0
            )

            sampler_logreward = logreward[indices]
            logreward = torch.cat([sampler_logreward, samples["logreward"]], dim=0)

        loss = self.compute_loss(trajectories, actions, dones, logreward)
        additional_metrics = self.env.compute_metrics(final_state, logreward)

        if loss is not None:
            self.train_loss(loss)
        self.train_logreward(sampler_logreward.mean())
        self.train_trajectory_length(trajectory_length.float().mean())

        if loss is not None:
            self.log(
                "train/loss",
                self.train_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        self.log(
            "train/logreward",
            self.train_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.replay_buffer is not None:
            self.log(
                "replay_buffer_size",
                float(len(self.replay_buffer)),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "replay_buffer_mean_logreward",
                self.replay_buffer.storage["logreward"].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.log(
            "train/trajectory_length",
            self.train_trajectory_length,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        for metric_name in additional_metrics:
            self.log(
                f"train/{metric_name}",
                additional_metrics[metric_name],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.hparams.chunk_algorithm is not None:
            if (
                self.current_epoch > 0
                and self.current_epoch % self.hparams.library_update_frequency == 0
                and batch_idx == 0
            ):
                self.update_library()
                if (
                    self.hparams.chunk_type == "replacement"
                    and self.replay_buffer is not None
                ):
                    self.refactor_replay_buffer()
        return loss

    @abstractmethod
    def validation_step(self, val_batch, batch_idx) -> Any:
        NotImplementedError

    def on_save_checkpoint(self, checkpoint):
        """Add the replay buffer to the checkpoint.
        Args:
            checkpoint (dict): Checkpoint dictionary.
        """
        checkpoint["replay_buffer"] = self.replay_buffer

    def on_load_checkpoint(self, checkpoint):
        """Load the replay buffer from the checkpoint.
        Args:
            checkpoint (dict): Checkpoint dictionary.
        """
        self.replay_buffer = checkpoint["replay_buffer"]
