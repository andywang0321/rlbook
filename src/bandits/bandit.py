import numpy as np
from .utils import argmax


class Bandit:
    def __init__(self, k: int, stationary: bool = False, value: float = 0) -> None:
        self.k: int = k
        self.stationary: bool = stationary
        if stationary:
            self.values: list[float] = [np.random.normal(value, 1) for _ in range(k)]
        else:
            q_star = np.random.normal(0, 1)
            self.values = [q_star for _ in range(k)]
        self.optimal_actions: list[int] = argmax(self.values)

    def __call__(self, action: int) -> float:
        reward = np.random.normal(self.values[action], 1.0)
        if not self.stationary:
            self.random_walk()
        return reward

    def random_walk(self) -> None:
        for i in range(self.k):
            self.values[i] += np.random.normal(0, 0.01)
        self.optimal_actions = argmax(self.values)
