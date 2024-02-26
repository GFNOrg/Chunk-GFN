from . import Scheduler


class ExponentialSchedule(Scheduler):
    def __init__(self, initial_value: float, factor: float):
        self.factor = factor
        self.initial_value = initial_value

    def step(self, epoch: int) -> float:
        value = self.initial_value * (self.factor**epoch)
        return value
