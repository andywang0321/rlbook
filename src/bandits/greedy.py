import numpy as np
from .utils import argmax


class Gambler:
    def __init__(
        self,
        k: int,
        epsilon: float,
        alpha: float | None = None,
        initial_values: list[float] | None = None,
    ) -> None:
        assert 0.0 <= epsilon <= 1.0, "epsilon needs to be in [0, 1]!"
        self.alpha: float | None = alpha
        if alpha is None:
            self.counts: list[int] = [0 for _ in range(k)]
        self.epsilon: float = epsilon
        self.k: int = k
        self.estimates: list[float] = initial_values if initial_values else [0.0] * k

    def __call__(self) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.k))
        return int(np.random.choice(argmax(self.estimates)))

    def step(self, action: int, reward: float) -> None:
        if self.alpha is not None:
            step_size = self.alpha
        else:
            self.counts[action] += 1
            step_size = 1 / self.counts[action]
        self.estimates[action] += step_size * (reward - self.estimates[action])
