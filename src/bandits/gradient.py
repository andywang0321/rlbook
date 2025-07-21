"""
Gradient Bandit Algorithm
"""

from .utils import softmax
import numpy as np
from numpy.typing import NDArray


class GradientGambler:
    def __init__(self, k: int, alpha: float, use_baseline: bool = True):
        self.k: int = k
        self.alpha: float = alpha
        self.preferences: NDArray[np.float64] = np.zeros(k, dtype=np.float64)
        self.probs: NDArray[np.float64] = np.full(k, 1.0 / k)
        self.baseline: float = 0.0
        self.use_baseline: bool = use_baseline
        self.t: int = 0

    def __call__(self) -> int:
        self.probs = softmax(self.preferences)
        return np.random.choice(self.k, p=self.probs)

    def step(self, action: int, reward: float) -> None:
        self.t += 1
        if self.use_baseline:
            self.baseline += (1 / self.t) * (reward - self.baseline)

        baseline = self.baseline if self.use_baseline else 0.0
        one_hot = np.zeros(self.k)
        one_hot[action] = 1

        self.preferences += self.alpha * (reward - baseline) * (one_hot - self.probs)

    @property
    def label(self) -> str:
        base = r"Gradient, $\alpha=" + f"{self.alpha}$"
        return base + ", baseline" if self.use_baseline else base + ", no baseline"
