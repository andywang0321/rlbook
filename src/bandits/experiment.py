import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from joblib import Parallel, delayed
from tqdm import trange
import copy
from .bandit import Bandit
from .greedy import Gambler


class Run:
    def __init__(self, bandit: Bandit, gambler: Gambler) -> None:
        self.bandit: Bandit = bandit
        self.gambler: Gambler = gambler

    def step(self) -> tuple[int, float]:
        action = self.gambler()
        reward = self.bandit(action)
        self.gambler.step(action, reward)
        return action, reward


def simulate_one_run(
    k: int, run_len: int, stationary: bool, gambler_templates: list[Gambler]
):
    bandit = Bandit(k, stationary)
    gamblers = [copy.deepcopy(g) for g in gambler_templates]
    runs = [Run(bandit, g) for g in gamblers]
    num_agents = len(runs)

    rewards = np.empty((num_agents, run_len))
    optimal = np.empty((num_agents, run_len), dtype=bool)

    for t in range(run_len):
        for i, run in enumerate(runs):
            action, reward = run.step()
            rewards[i, t] = reward
            optimal[i, t] = action in bandit.optimal_actions

    return rewards, optimal


def simulate_parallel(
    k: int,
    run_len: int,
    num_runs: int,
    stationary: bool,
    gambler_templates: list[Gambler],
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run bandit simulations in parallel using deepcopy to clone agents.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_one_run)(k, run_len, stationary, gambler_templates)
        for _ in trange(num_runs, desc="Stationary" if stationary else "Nonstationary")
    )

    rewards_all: np.ndarray = np.stack([res[0] for res in results], axis=1)
    optimal_all: np.ndarray = np.stack([res[1] for res in results], axis=1)
    return rewards_all, optimal_all


def plot_results(
    rewards_all: np.ndarray,
    optimal_all: np.ndarray,
    labels: list[str],
    title_prefix: str = "",
    show_rewards: bool = True,
    show_optimal: bool = True,
    axes: list[Axes] | None = None,
):
    num_agents = rewards_all.shape[0]

    if axes is None:
        n_rows = int(show_rewards) + int(show_optimal)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows))
        if n_rows == 1:
            axes = [axes]

    ax_idx = 0
    if show_rewards:
        for i in range(num_agents):
            mean_reward = rewards_all[i].mean(axis=0)
            axes[ax_idx].plot(mean_reward, label=labels[i])
        axes[ax_idx].set_title(f"{title_prefix} Average Rewards")
        axes[ax_idx].set_xlabel("Steps")
        axes[ax_idx].set_ylabel("Reward")
        axes[ax_idx].legend()
        ax_idx += 1

    if show_optimal:
        for i in range(num_agents):
            mean_optimal = optimal_all[i].mean(axis=0)
            axes[ax_idx].plot(mean_optimal, label=labels[i])
        axes[ax_idx].set_title(f"{title_prefix} Optimal Action Proportion")
        axes[ax_idx].set_xlabel("Steps")
        axes[ax_idx].set_ylabel("Proportion Optimal")
        axes[ax_idx].set_ylim(0, 1)
        axes[ax_idx].legend()

    if axes is None:
        plt.tight_layout()
        plt.show()


def run_experiment(
    k: int,
    run_len: int,
    num_runs: int,
    gambler_templates: list[Gambler],
    labels: list[str],
    show_stationary: bool = True,
    show_nonstationary: bool = True,
    show_rewards: bool = True,
    show_optimal: bool = True,
    n_jobs: int = -1,
):
    """
    Run experiment with agent instances passed directly.

    gambler_templates: list of Gambler objects (will be deepcopy'd inside each run)
    """
    cols = int(show_stationary) + int(show_nonstationary)
    rows = int(show_rewards) + int(show_optimal)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    col = 0
    if show_stationary:
        rewards, optimal = simulate_parallel(
            k,
            run_len,
            num_runs,
            stationary=True,
            gambler_templates=gambler_templates,
            n_jobs=n_jobs,
        )
        plot_results(
            rewards,
            optimal,
            labels,
            title_prefix="Stationary",
            show_rewards=show_rewards,
            show_optimal=show_optimal,
            axes=axes[:, col],
        )
        col += 1

    if show_nonstationary:
        rewards, optimal = simulate_parallel(
            k,
            run_len,
            num_runs,
            stationary=False,
            gambler_templates=gambler_templates,
            n_jobs=n_jobs,
        )
        plot_results(
            rewards,
            optimal,
            labels,
            title_prefix="Nonstationary",
            show_rewards=show_rewards,
            show_optimal=show_optimal,
            axes=axes[:, col],
        )

    plt.tight_layout()
    plt.show()
