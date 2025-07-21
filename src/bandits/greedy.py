"""
Epsilon-Greedy Action Selection Method
"""

import numpy as np
from numpy.typing import NDArray
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
        if alpha:
            assert 0.0 < alpha <= 1.0, "alpha needs to be in (0, 1]!"
        self.alpha: float | None = alpha
        self.counts: NDArray[np.intp] = np.zeros(k, dtype=np.intp)
        self.epsilon: float = epsilon
        self.k: int = k
        if initial_values:
            assert len(initial_values) == k, "Length of initial_values must match k!"
            self.estimates: NDArray[np.float64] = np.array(initial_values)
        else:
            self.estimates = np.zeros(k)

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

    @property
    def label(self) -> str:
        strategy = f"$\\epsilon={self.epsilon}"
        if self.alpha is None:
            strategy += r", \alpha=\frac{1}{n}$"
        else:
            strategy += f", \\alpha={self.alpha}$"
        return r"$\epsilon$-greedy " + strategy
