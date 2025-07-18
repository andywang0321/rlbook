"""
Exercise 2.5
Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary problems. Use a modified version of the 10-armed testbed in which all the $q_*(a)$ start out equal and then take independent random walks (say by adding a normally distributed increment with mean zero and standard deviation 0.01 to all the $q_*(a)$ on each step). Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally computed, and another action-value method using a constant step-size parameter, $\\alpha = 0.1$. Use $\\epsilon = 0.1$ and longer runs, say of 10,000 steps.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from joblib import Parallel, delayed


def argmax(x: list[float]) -> list[int]:
    max_val = max(x)
    return [i for i, val in enumerate(x) if val == max_val]


class Bandit:
    def __init__(self, k: int, stationary: bool = False) -> None:
        self.k: int = k
        self.stationary: bool = stationary
        if stationary:
            self.values: list[float] = [np.random.normal(0, 1) for _ in range(k)]
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


class Run:
    def __init__(self, bandit: Bandit, gambler: Gambler) -> None:
        self.bandit: Bandit = bandit
        self.gambler: Gambler = gambler

    def step(self) -> tuple[int, float]:
        action = self.gambler()
        reward = self.bandit(action)
        self.gambler.step(action, reward)
        return action, reward


def simulate_one_run(k: int, run_len: int, stationary: bool):
    bandit = Bandit(k, stationary)
    run1 = Run(bandit, Gambler(k, epsilon=0.1, alpha=None))
    run2 = Run(bandit, Gambler(k, epsilon=0.1, alpha=0.1))

    rewards1 = np.empty(run_len)
    rewards2 = np.empty(run_len)
    optimal1 = np.empty(run_len, dtype=bool)
    optimal2 = np.empty(run_len, dtype=bool)

    for t in range(run_len):
        a1, r1 = run1.step()
        a2, r2 = run2.step()
        rewards1[t] = r1
        rewards2[t] = r2
        optimal1[t] = a1 in bandit.optimal_actions
        optimal2[t] = a2 in bandit.optimal_actions

    return rewards1, rewards2, optimal1, optimal2


def simulate_parallel(stationary: bool, k: int, run_len: int, num_runs: int, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_one_run)(k, run_len, stationary)
        for _ in trange(num_runs, desc="Stationary" if stationary else "Nonstationary")
    )

    rewards1, rewards2, optimal1, optimal2 = zip(*results)
    return (
        np.stack(rewards1),
        np.stack(rewards2),
        np.stack(optimal1),
        np.stack(optimal2),
    )


# --- Parameters ---
k = 10
num_runs = 3000
run_len = 10000

# --- Run Experiments ---
results_stationary = simulate_parallel(True, k, run_len, num_runs)
results_nonstationary = simulate_parallel(False, k, run_len, num_runs)

# --- Plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

titles = ["Stationary Bandit", "Nonstationary Bandit"]
labels = [r"$\epsilon=0.1, \alpha=\frac{1}{n}$", r"$\epsilon=0.1, \alpha=0.1$"]
colors = ["C0", "C1"]

for col, results in enumerate([results_stationary, results_nonstationary]):
    rewards1, rewards2, opt1, opt2 = results

    axes[0, col].plot(np.mean(rewards1, axis=0), label=labels[0], color=colors[0])
    axes[0, col].plot(np.mean(rewards2, axis=0), label=labels[1], color=colors[1])
    axes[0, col].set_title(f"Average Rewards ({titles[col]})")
    axes[0, col].set_xlabel("Steps")
    axes[0, col].set_ylabel("Reward")
    axes[0, col].legend()

    axes[1, col].plot(np.mean(opt1, axis=0), label=labels[0], color=colors[0])
    axes[1, col].plot(np.mean(opt2, axis=0), label=labels[1], color=colors[1])
    axes[1, col].set_title(f"Optimal Action Proportion ({titles[col]})")
    axes[1, col].set_xlabel("Steps")
    axes[1, col].set_ylabel("Proportion Optimal")
    axes[1, col].set_ylim(0, 1)
    axes[1, col].legend()

plt.tight_layout()
plt.show()
