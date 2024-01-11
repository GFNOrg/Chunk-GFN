from abc import ABC, abstractmethod


class Scheduler(ABC):
    @abstractmethod
    def step(self, epoch: int) -> float:
        NotImplementedError
