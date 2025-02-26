from typing import Any, Tuple

import torch
from einops import rearrange, repeat
from torch import nn
from torch.distributions import Bernoulli, Categorical

from chunkgfn.algo.sampler_base import BaseSampler
from chunkgfn.algo.utils import has_trainable_parameters
from chunkgfn.schedulers import Scheduler

from ..constants import EPS, NEGATIVE_INFINITY


class OptionCritic(BaseSampler):
    """Option Critic module."""

    def __init__(
        self,
        forward_policy: nn.Module,
        action_embedder: nn.Module,
        critic: nn.Module,
        beta: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        option_epsilon_scheduler: Scheduler | None = None,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)
        super().__init__(
            forward_policy,
            action_embedder,
            optimizer,
            scheduler,
            epsilon_scheduler=option_epsilon_scheduler,
            temperature_scheduler=None,
            replay_buffer=None,
            **kwargs,
        )
        self.critic = critic
        self.beta = beta  # This is the termination policy
        # https://github.com/lweitkamp/option-critic-pytorch/blob/master/option_critic.py
        self.options_W = nn.Parameter(
            torch.randn(
                self.hparams.num_options,
                forward_policy.action_embedding_dim,
                forward_policy.action_embedding_dim,
            )
        )
        self.options_b = nn.Parameter(
            torch.zeros(self.hparams.num_options, forward_policy.action_embedding_dim)
        )

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

        if self.critic is not None and has_trainable_parameters(self.critic):
            params.append(
                {
                    "params": self.critic.parameters(),
                    "lr": self.hparams.critic_lr,
                }
            )

        params.append(
            {
                "params": self.options_W,
                "lr": self.hparams.option_lr,
            }
        )
        params.append(
            {
                "params": self.options_b,
                "lr": self.hparams.option_lr,
            }
        )
        params.append(
            {
                "params": self.beta.parameters(),
                "lr": self.hparams.beta_lr,
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

    def get_forward_logits(
        self, state: torch.Tensor, option: torch.Tensor
    ) -> torch.Tensor:
        """Get the forward logits for the given state.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): State.
            option (torch.Tensor[batch_size]): Batch of options.
        Return:
            logits (torch.Tensor[batch_size, n_actions]): Forward logits.
        """
        processed = self.env.preprocess_states(state)
        if isinstance(processed, tuple):
            action_embedding = self.forward_policy(*processed)
        else:
            action_embedding = self.forward_policy(processed)

        # Select the option specific weights and biases
        W = self.options_W[option]
        b = self.options_b[option]

        # print(action_embedding.shape, W.shape, b.shape)
        action_embedding = torch.einsum("bd, bdc -> bc", action_embedding, W) + b

        dim = action_embedding.shape[-1]
        library_embeddings = self.get_library_embeddings()
        logits = torch.einsum("bd, nd -> bn", action_embedding, library_embeddings) / (
            dim**0.5
        )  # Same as in softmax
        return logits

    def get_option_termination(self, state: torch.Tensor, option: torch.Tensor):
        """Get the option termination mask for the given state.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): State.
            option (torch.Tensor[batch_size]): Batch of options.
        Return:
            option_termination (torch.Tensor[batch_size, num_options]): Option termination mask.
            greedy_options (torch.Tensor[batch_size]): Greedy options.
        """
        processed = self.env.preprocess_states(state)

        if isinstance(processed, tuple):
            termination = self.beta(*processed)
        else:
            termination = self.beta(processed)

        termination = torch.sigmoid(termination)
        termination = termination[torch.arange(state.shape[0]), option]
        option_termination = Bernoulli(termination).sample()

        if isinstance(processed, tuple):
            greedy_options = self.critic(*processed).argmax(dim=-1)
        else:
            greedy_options = self.critic(processed).argmax(dim=-1)

        return option_termination, greedy_options

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
            epsilon (float|None): Epsilon value for epsilon-soft selection of the option.
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
        options = []
        done = torch.zeros((bs), dtype=torch.bool, device=self.device)
        trajectory_length = torch.zeros(
            (bs), dtype=torch.long, device=self.device
        )  # This tracks the length of trajectory for each sample in the batch

        current_options = torch.randint(
            self.hparams.num_options, (bs,), device=self.device
        )  # Randomly sample an option for each sample in the batch
        greedy_options = current_options.clone()
        option_termination = torch.ones(
            (bs, self.hparams.num_options), dtype=torch.bool, device=self.device
        )
        while not done.all():
            # Sample a new option in an epsilon-soft way if the current option has terminated

            new_options = torch.randint(
                self.hparams.num_options, (bs,), device=self.device
            )
            if epsilon is not None:
                new_options = torch.where(
                    torch.rand((bs,), device=self.device) <= epsilon,
                    new_options,
                    greedy_options,
                )
            else:
                new_options = greedy_options

            current_options = torch.where(
                option_termination.any(dim=-1), new_options, current_options
            )

            logit_pf = self.get_forward_logits(state, current_options)
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

            option_termination, greedy_options = self.get_option_termination(
                new_state, current_options
            )
            actions.append(act)
            trajectories.append(state)
            dones.append(done.clone())
            options.append(current_options)

            state = new_state.clone()

        trajectories.append(state)
        dones.append(torch.ones((bs), dtype=torch.bool, device=self.device))
        trajectories = torch.stack(trajectories, dim=1)
        actions = torch.stack(actions, dim=1)
        dones = torch.stack(dones, dim=1)
        options = torch.stack(options, dim=1)

        return trajectories, actions, dones, options, state, trajectory_length

    def sample(
        self,
        batch: torch.Tensor,
        train: bool = True,
        epsilon: float = 0.0,
        temperature: float = 0.0,
        calculate_logreward: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch

        trajectories, actions, dones, options, final_state, trajectory_length = (
            self.forward(
                x.shape[0], train=train, epsilon=epsilon, temperature=temperature
            )
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
            options,
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

        (
            x,
            trajectories,
            actions,
            dones,
            options,
            final_state,
            logreward,
            trajectory_length,
        ) = self.sample(
            train_batch,
            train=True,
            epsilon=epsilon,
            temperature=temperature,
        )

        sampler_logreward = logreward

        loss = self.compute_loss(trajectories, actions, options, dones, logreward)
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

        return loss

    def compute_loss(self, trajectories, actions, options, dones, logreward):
        """Compute the loss for the model.
        Args:
            trajectories (torch.Tensor[batch_size, trajectory_length, *state_shape]): Trajectories for each sample in the batch.
            actions (torch.Tensor[batch_size, trajectory_length]): Actions for each sample in the batch.
            options (torch.Tensor[batch_size, trajectory_length]): Options for each sample in the batch.
            dones (torch.Tensor[batch_size, trajectory_length]): Whether the trajectory is done or not.
            logreward (torch.Tensor[batch_size]): Log reward.
        """

        # We don't compute the value and policy loss for the final state
        trajectories = trajectories[:, :-1]
        dones = dones[:, :-1]

        states = rearrange(trajectories, "b t ... -> (b t) ...")
        options = rearrange(options, "b t -> (b t)")
        # Since reward only given at the end, return=reward
        returns = logreward.repeat_interleave(trajectories.shape[1]).exp()
        processed = self.env.preprocess_states(states)

        if isinstance(processed, tuple):
            values = self.critic(*processed)
        else:
            values = self.critic(processed)

        values_max = values.max(dim=-1)[0]

        values = values[
            torch.arange(states.shape[0]), options.flatten()
        ]  # Get the value for the option taken

        advantage = returns - values

        logits = self.get_forward_logits(states, options)
        forward_mask = self.env.get_forward_mask(states)
        logits = torch.where(
            forward_mask,
            logits,
            torch.tensor(NEGATIVE_INFINITY).to(logits),
        )

        value_loss = advantage.pow(2)
        logp = Categorical(logits=logits).log_prob(
            rearrange(actions, "b t ... -> (b t) ...")
        )

        processed = self.env.preprocess_states(states)

        if isinstance(processed, tuple):
            termination = self.beta(*processed)
        else:
            termination = self.beta(processed)

        option_term_prob = termination[torch.arange(states.shape[0]), options]
        termination_loss = option_term_prob * (
            (values - values_max).detach() + self.hparams.termination_reg
        )

        policy_loss = -advantage.detach() * logp
        policy_loss = policy_loss + termination_loss
        entropy_loss = Categorical(logits=logits).entropy()

        loss = rearrange(
            (value_loss + policy_loss - self.hparams.entropy_coeff * entropy_loss),
            "(b t) -> b t",
            b=trajectories.shape[0],
        )
        loss = torch.where(dones, 0, loss)
        loss = torch.clamp(loss.sum(1), max=5000, min=-5000)
        loss = loss.mean()

        return loss

    def validation_step(self, val_batch, batch_idx) -> Any:
        final_state, logreward = val_batch
        (
            x,
            trajectories,
            actions,
            dones,
            options,
            final_state,
            logreward,
            trajectory_length,
        ) = self.sample(final_state, train=False, epsilon=None, temperature=None)
        loss = self.compute_loss(trajectories, actions, options, dones, logreward)

        self.val_loss(loss)
        self.val_logreward(logreward.mean())

        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)
        self.log(
            "val/logreward",
            self.val_logreward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        # Get on-policy samples from the GFN
        dummy_batch = torch.arange(self.hparams.n_onpolicy_samples).to(self.device)
        _, _, actions, dones, options, final_state, _, trajectory_length = self.sample(
            dummy_batch,
            train=False,
            epsilon=None,
            temperature=None,
            calculate_logreward=False,
        )
        torch.save(
            {
                "actions": actions,
                "dones": dones,
                "final_state": final_state,
                "trajectory_length": trajectory_length,
                "epoch": self.current_epoch,
            },
            f"{self.trainer.log_dir}/on_policy_samples_{self.current_epoch}.pt",
        )

        # Save the library and frequency of use at each epoch
        action_frequency = [
            [freq, action]
            for (freq, action) in zip(
                self.env.action_frequency,
                self.env.actions,
            )
        ]
        torch.save(
            action_frequency, f"{self.trainer.log_dir}/library_{self.current_epoch}.pt"
        )
