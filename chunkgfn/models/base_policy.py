from torch import nn

from chunkgfn.environment.base_module import BaseUnConditionalEnvironmentModule


class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self._environment = None

    def set_environment(self, env: BaseUnConditionalEnvironmentModule) -> None:
        self._environment = env

    def get_environment(self) -> BaseUnConditionalEnvironmentModule:
        return self._environment

    def __getstate__(self):
        # Remove the environment before pickling
        state = self.__dict__.copy()
        state["_environment"] = None
        return state

    def __setstate__(self, state):
        # Restore the state without the environment
        self.__dict__.update(state)

    def forward(self, x):
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses should implement this!")
