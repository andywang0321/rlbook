"""
Upper-Confidence-Bound Action Selection
"""

import numpy as np
from numpy.typing import NDArray
from .utils import argmax
from .greedy import Gambler


class UCBGambler(Gambler):
    def __init__(
        self,
        k: int,
        c: float,
        epsilon: float = 0,
        alpha: float | None = None,
        initial_values: list[float] | None = None,
    ) -> None:
        super().__init__(k, epsilon, alpha, initial_values)
        assert 0.0 < c, "c needs to be > 0!"
        self.c: float = c
        self.t: int = 0

    def __call__(self) -> int:
        self.t += 1
        if any(self.counts == 0):
            actions: NDArray[np.intp] = np.where(self.counts == 0)[0]
        else:
            uncertainty: np.ndarray = self.c * np.sqrt(np.log(self.t) / self.counts)
            actions = argmax((self.estimates + uncertainty))
        return np.random.choice(actions)

    @property
    def label(self) -> str:
        strategy = f"$\\epsilon={self.epsilon}"
        if self.alpha is None:
            strategy += r", \alpha=\frac{1}{n}$"
        else:
            strategy += f", \\alpha={self.alpha}$"
        strategy += f", $c={self.c}$"
        return "UCB " + strategy
