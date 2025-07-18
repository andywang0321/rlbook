"""
Exercise 2.5
Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary problems. Use a modified version of the 10-armed testbed in which all the $q_*(a)$ start out equal and then take independent random walks (say by adding a normally distributed increment with mean zero and standard deviation 0.01 to all the $q_*(a)$ on each step). Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally computed, and another action-value method using a constant step-size parameter, $\\alpha = 0.1$. Use $\\epsilon = 0.1$ and longer runs, say of 10,000 steps.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def argmax(x: list[float]) -> list[int]:
    max_val: float = max(x)
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
        if self.stationary:
            reward: float = np.random.normal(self.values[action], 1.0)
        else:
            reward = np.random.normal(self.values[action], 1.0)
            self.random_walk()
        return reward

    def random_walk(self) -> None:
        assert not self.stationary, "Stationary bandit should not random walk!"
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
        if alpha:
            assert 0.0 <= alpha <= 1.0, "alpha needs to be in [0, 1]!"
        else:
            self.counts: list[int] = [0 for _ in range(k)]

        self.epsilon: float = epsilon
        self.k: int = k

        if initial_values:
            self.estimates: list[float] = initial_values
        else:
            self.estimates = [0.0 for _ in range(k)]

    def __call__(self) -> int:
        if np.random.rand() < self.epsilon:
            action: int = np.random.choice(self.k)
        else:
            action = np.random.choice(argmax(self.estimates))
        return int(action)

    def step(self, action: int, reward: float) -> None:
        if self.alpha:
            step_size = self.alpha
        else:
            self.counts[action] += 1
            step_size: float = 1 / self.counts[action]
        self.estimates[action] += step_size * (reward - self.estimates[action])


class Run:
    def __init__(self, bandit: Bandit, gambler: Gambler) -> None:
        self.gambler: Gambler = gambler
        self.bandit: Bandit = bandit

    def step(self) -> tuple[int, float]:
        action: int = self.gambler()
        reward: float = self.bandit(action)
        self.gambler.step(action, reward)
        return action, reward


k = 10
num_runs = 2000
run_len = 10000


def simulate(stationary):
    rewards_alpha_none, rewards_alpha_point1 = [], []
    optimal_alpha_none, optimal_alpha_point1 = [], []

    for _ in trange(num_runs, desc="Stationary" if stationary else "Nonstationary"):
        bandit = Bandit(k, stationary=stationary)

        run_none = Run(bandit, Gambler(k, epsilon=0.1, alpha=None))
        run_point1 = Run(bandit, Gambler(k, epsilon=0.1, alpha=0.1))

        r_none, r_point1 = [], []
        o_none, o_point1 = [], []

        for _ in range(run_len):
            a1, rew1 = run_none.step()
            a2, rew2 = run_point1.step()

            r_none.append(rew1)
            r_point1.append(rew2)

            o_none.append(a1 in bandit.optimal_actions)
            o_point1.append(a2 in bandit.optimal_actions)

        rewards_alpha_none.append(r_none)
        rewards_alpha_point1.append(r_point1)

        optimal_alpha_none.append(o_none)
        optimal_alpha_point1.append(o_point1)

    return (
        rewards_alpha_none,
        rewards_alpha_point1,
        optimal_alpha_none,
        optimal_alpha_point1,
    )


# Run both experiments
results_stationary = simulate(stationary=True)
results_nonstationary = simulate(stationary=False)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

titles = ["Stationary Bandit", "Nonstationary Bandit"]
labels = [r"$\epsilon=0.1$", r"$\epsilon=0.1, \alpha=0.1$"]
colors = ["C0", "C1"]

for col, results in enumerate([results_stationary, results_nonstationary]):
    rewards1, rewards2, opt1, opt2 = results

    # Rewards
    axes[0, col].plot(np.mean(rewards1, axis=0), label=labels[0], color=colors[0])
    axes[0, col].plot(np.mean(rewards2, axis=0), label=labels[1], color=colors[1])
    axes[0, col].set_title(f"Average Rewards ({titles[col]})")
    axes[0, col].set_xlabel("Steps")
    axes[0, col].set_ylabel("Reward")
    axes[0, col].legend()

    # Optimal action proportions
    axes[1, col].plot(np.mean(opt1, axis=0), label=labels[0], color=colors[0])
    axes[1, col].plot(np.mean(opt2, axis=0), label=labels[1], color=colors[1])
    axes[1, col].set_title(f"Optimal Action Proportion ({titles[col]})")
    axes[1, col].set_xlabel("Steps")
    axes[1, col].set_ylabel("Proportion Optimal")
    axes[1, col].set_ylim(0, 1)
    axes[1, col].legend()

plt.tight_layout()
plt.show()

"""
k = 10
stationary = True
num_runs = 2000
run_len = 1000

rewards_list1: list[list[float]] = []
rewards_list2: list[list[float]] = []
rewards_list3: list[list[float]] = []
rewards_list4: list[list[float]] = []

optimal_actions_list1: list[list[bool]] = []
optimal_actions_list2: list[list[bool]] = []
optimal_actions_list3: list[list[bool]] = []
optimal_actions_list4: list[list[bool]] = []

for _ in trange(num_runs):
    bandit = Bandit(k, stationary)

    gambler1 = Gambler(k, epsilon=0)
    run1 = Run(bandit, gambler1)

    gambler2 = Gambler(k, epsilon=0.1)
    run2 = Run(bandit, gambler2)

    gambler3 = Gambler(k, epsilon=0.01)
    run3 = Run(bandit, gambler3)

    gambler4 = Gambler(k, epsilon=0.1, alpha=0.1)
    run4 = Run(bandit, gambler4)

    rewards1: list[float] = []
    rewards2: list[float] = []
    rewards3: list[float] = []
    rewards4: list[float] = []

    optimal_action1: list[bool] = []
    optimal_action2: list[bool] = []
    optimal_action3: list[bool] = []
    optimal_action4: list[bool] = []

    for _ in range(run_len):
        a1, r1 = run1.step()
        a2, r2 = run2.step()
        a3, r3 = run3.step()
        a4, r4 = run4.step()

        rewards1.append(r1)
        rewards2.append(r2)
        rewards3.append(r3)
        rewards4.append(r4)

        optimal_action1.append(a1 in bandit.optimal_actions)
        optimal_action2.append(a2 in bandit.optimal_actions)
        optimal_action3.append(a3 in bandit.optimal_actions)
        optimal_action4.append(a4 in bandit.optimal_actions)

    rewards_list1.append(rewards1)
    rewards_list2.append(rewards2)
    rewards_list3.append(rewards3)
    rewards_list4.append(rewards4)

    optimal_actions_list1.append(optimal_action1)
    optimal_actions_list2.append(optimal_action2)
    optimal_actions_list3.append(optimal_action3)
    optimal_actions_list4.append(optimal_action4)

fig, axes = plt.subplots(2, 1, figsize=(12, 12))

axes[0].plot(np.array(rewards_list1).mean(axis=0), label="$\\epsilon = 0$ (greedy)")
axes[0].plot(np.array(rewards_list2).mean(axis=0), label="$\\epsilon = 0.1$")
axes[0].plot(np.array(rewards_list3).mean(axis=0), label="$\\epsilon = 0.01$")
axes[0].plot(
    np.array(rewards_list4).mean(axis=0), label="$\\epsilon = 0.1, \\alpha = 0.1$"
)
axes[0].set_xlabel("Steps")
axes[0].set_ylabel("Average reward")
axes[0].legend()

axes[1].set_ylim(0, 1)
axes[1].plot(
    np.array(optimal_actions_list1).mean(axis=0), label="$\\epsilon = 0$ (greedy)"
)
axes[1].plot(np.array(optimal_actions_list2).mean(axis=0), label="$\\epsilon = 0.1$")
axes[1].plot(np.array(optimal_actions_list3).mean(axis=0), label="$\\epsilon = 0.01$")
axes[1].plot(
    np.array(optimal_actions_list4).mean(axis=0),
    label="$\\epsilon = 0.1, \\alpha = 0.1$",
)
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Prop optimal action")
axes[1].legend()

plt.tight_layout()
plt.show()
"""
