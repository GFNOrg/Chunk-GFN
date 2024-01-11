from . import Scheduler


class LinearSchedule(Scheduler):
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        max_epochs: int | None = None,
        rate: float | None = 0.5,
    ):
        assert not (
            max_epochs is None and rate is None
        ), "You can't set max_epochs and rate to None at the same time."
        self.rate = rate
        self.initial_value = initial_value
        self.final_value = final_value
        self.max_epochs = max_epochs

    def step(self, epoch: int) -> float:
        if self.max_epochs is not None:
            value = (
                self.initial_value
                + (self.final_value - self.initial_value) * epoch / self.max_epochs
            )
        else:
            if self.initial_value > self.final_value:
                value = max(self.final_value, self.initial_value - self.rate * epoch)
            else:
                value = min(self.final_value, self.initial_value + self.rate * epoch)
        return value
