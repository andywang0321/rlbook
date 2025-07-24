"""
Parameter Study
"""

from src.bandits import Gambler, UCBGambler, GradientGambler, simulate_parallel
import numpy as np
import pandas as pd
from numpy.typing import NDArray
import matplotlib.pyplot as plt

k: int = 10
run_len: int = 200_000
num_runs: int = 2000
avg_over: int = 100_000
stationary: bool = False
show_rewards: bool = True
show_optimal: bool = True
data_save_path: str = "data/ex_0211"


def save_data(
    name: str,
    parameters: NDArray[np.float64],
    rewards_all: NDArray[np.float64],
    optimal_all: NDArray[np.float64],
) -> None:
    """
    Save average rewards and optimal action proportions as CSV files.

    Args:
        name: base filename prefix (e.g. 'epsilon_greedy')
        parameters: shape (num_agents,), e.g. [0.0, 0.1, 0.2]
        rewards: shape (num_agents, num_runs, run_len)
        optimal: shape (num_agents, num_runs, run_len)
    """
    path: str = data_save_path + f"/{name}"
    # Compute means over runs
    mean_rewards = rewards_all.mean(axis=1)  # shape: (num_agents, run_len)
    mean_optimal = optimal_all.mean(axis=1)  # shape: (num_agents, run_len)

    # Convert to DataFrames: shape (run_len, num_agents)
    reward_df = pd.DataFrame(mean_rewards.T, columns=[f"{p:.4f}" for p in parameters])
    optimal_df = pd.DataFrame(mean_optimal.T, columns=[f"{p:.4f}" for p in parameters])

    # Save
    reward_df.to_csv(f"{path}_rewards.csv", index_label="step")
    optimal_df.to_csv(f"{path}_optimal.csv", index_label="step")

    print(f"✅ Saved: {path}_rewards.csv and {path}_optimal.csv")


epsilons: NDArray[np.float64] = np.logspace(-7, -2, num=6, base=2)
alphas: NDArray[np.float64] = np.logspace(-5, 2, num=8, base=2)
confidences: NDArray[np.float64] = np.logspace(-4, 2, num=7, base=2)
Q_0s: NDArray[np.float64] = np.logspace(-2, 2, num=5, base=2)

eps_greedy_templates: list[Gambler] = [Gambler(k=k, epsilon=e) for e in epsilons]
gradient_templates: list[GradientGambler] = [GradientGambler(k=k, alpha=a) for a in alphas]
ucb_templates: list[UCBGambler] = [UCBGambler(k=k, c=c) for c in confidences]
optimist_templates: list[Gambler] = [Gambler(k=k, epsilon=0, alpha=0.1, initial_values=[Q_0] * k) for Q_0 in Q_0s]
const_step_templates: list[Gambler] = [Gambler(k=k, epsilon=e, alpha=0.1) for e in epsilons]

# print("Simulating eps-greedy...")
# eps_greedy_rewards_all, eps_greedy_optimal_all = simulate_parallel(
#     k=k, run_len=run_len, num_runs=num_runs, stationary=stationary, gambler_templates=eps_greedy_templates
# )
# save_data("eps_greedy", epsilons, eps_greedy_rewards_all, eps_greedy_optimal_all)
# eps_greedy_rewards: NDArray[np.float64] = eps_greedy_rewards_all[:, :, -avg_over:].mean(axis=(1, 2))
# eps_greedy_optimal: NDArray[np.float64] = eps_greedy_optimal_all[:, :, -avg_over:].mean(axis=(1, 2))

# load eps_greedy data
eps_greedy_rewards_all = pd.read_csv("data/ex_0211/eps_greedy_rewards.csv").drop("step", axis=1).values
eps_greedy_optimal_all = pd.read_csv("data/ex_0211/eps_greedy_optimal.csv").drop("step", axis=1).values
eps_greedy_rewards: NDArray[np.float64] = eps_greedy_rewards_all[-avg_over:].mean(axis=0)
eps_greedy_optimal: NDArray[np.float64] = eps_greedy_optimal_all[-avg_over:].mean(axis=0)

# print("Simulating gradient bandit...")
# gradient_rewards_all, gradient_optimal_all = simulate_parallel(
#     k=k, run_len=run_len, num_runs=num_runs, stationary=stationary, gambler_templates=gradient_templates
# )
# save_data("gradient_bandit", alphas, gradient_rewards_all, gradient_optimal_all)
# gradient_rewards: NDArray[np.float64] = gradient_rewards_all[:, :, -avg_over:].mean(axis=(1, 2))
# gradient_optimal: NDArray[np.float64] = gradient_optimal_all[:, :, -avg_over:].mean(axis=(1, 2))

# load gradient data
gradient_rewards_all = pd.read_csv("data/ex_0211/gradient_bandit_rewards.csv").drop("step", axis=1).values
gradient_optimal_all = pd.read_csv("data/ex_0211/gradient_bandit_optimal.csv").drop("step", axis=1).values
gradient_rewards: NDArray[np.float64] = gradient_rewards_all[-avg_over:].mean(axis=0)
gradient_optimal: NDArray[np.float64] = gradient_optimal_all[-avg_over:].mean(axis=0)

# print("Simulating UCB...")
# ucb_rewards_all, ucb_optimal_all = simulate_parallel(
#     k=k, run_len=run_len, num_runs=num_runs, stationary=stationary, gambler_templates=ucb_templates
# )
# save_data("ucb", confidences, ucb_rewards_all, ucb_optimal_all)
# ucb_rewards: NDArray[np.float64] = ucb_rewards_all[:, :, -avg_over:].mean(axis=(1, 2))
# ucb_optimal: NDArray[np.float64] = ucb_optimal_all[:, :, -avg_over:].mean(axis=(1, 2))

# load ucb data
ucb_rewards_all = pd.read_csv("data/ex_0211/ucb_rewards.csv").drop("step", axis=1).values
ucb_optimal_all = pd.read_csv("data/ex_0211/ucb_optimal.csv").drop("step", axis=1).values
ucb_rewards: NDArray[np.float64] = ucb_rewards_all[-avg_over:].mean(axis=0)
ucb_optimal: NDArray[np.float64] = ucb_optimal_all[-avg_over:].mean(axis=0)

# print("Simulating optimistic initial values...")
# optimist_rewards_all, optimist_optimal_all = simulate_parallel(
#     k=k, run_len=run_len, num_runs=num_runs, stationary=stationary, gambler_templates=optimist_templates
# )
# save_data("optimist", Q_0s, optimist_rewards_all, optimist_optimal_all)
# optimist_rewards: NDArray[np.float64] = optimist_rewards_all[:, :, -avg_over:].mean(axis=(1, 2))
# optimist_optimal: NDArray[np.float64] = optimist_optimal_all[:, :, -avg_over:].mean(axis=(1, 2))

# load optimist data
optimist_rewards_all = pd.read_csv("data/ex_0211/optimist_rewards.csv").drop("step", axis=1).values
optimist_optimal_all = pd.read_csv("data/ex_0211/optimist_optimal.csv").drop("step", axis=1).values
optimist_rewards: NDArray[np.float64] = optimist_rewards_all[-avg_over:].mean(axis=0)
optimist_optimal: NDArray[np.float64] = optimist_optimal_all[-avg_over:].mean(axis=0)

# print("Simulating constant step size...")
# const_step_rewards_all, const_step_optimal_all = simulate_parallel(
#     k=k, run_len=run_len, num_runs=num_runs, stationary=stationary, gambler_templates=const_step_templates
# )
# save_data("constant_step_size", epsilons, const_step_rewards_all, const_step_optimal_all)
# const_step_rewards: NDArray[np.float64] = const_step_rewards_all[:, :, -avg_over:].mean(axis=(1, 2))
# const_step_optimal: NDArray[np.float64] = const_step_optimal_all[:, :, -avg_over:].mean(axis=(1, 2))

# load constant_step_size data
const_step_rewards_all = pd.read_csv("data/ex_0211/constant_step_size_rewards.csv").drop("step", axis=1).values
const_step_optimal_all = pd.read_csv("data/ex_0211/constant_step_size_optimal.csv").drop("step", axis=1).values
const_step_rewards: NDArray[np.float64] = const_step_rewards_all[-avg_over:].mean(axis=0)
const_step_optimal: NDArray[np.float64] = const_step_optimal_all[-avg_over:].mean(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(7.3, 8.4))

if show_rewards:
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel(r"$\epsilon$   $\alpha$   $c$   $Q_0$")
    axes[0].set_ylabel(f"Average reward over {avg_over} steps")
    axes[0].set_title("Stationary" if stationary else "Nonstationary")
    axes[0].plot(epsilons, eps_greedy_rewards, label=r"$\epsilon$-greedy", color="red")
    axes[0].plot(alphas, gradient_rewards, label="gradient bandit", color="green")
    axes[0].plot(confidences, ucb_rewards, label="UCB", color="purple")
    axes[0].plot(Q_0s, optimist_rewards, label=r"greedy with optimistic initialization $\alpha=0.1$", color="black")
    axes[0].plot(
        epsilons, const_step_rewards, label=r"constant step size $\epsilon$-greedy with $\alpha=0.1$", color="blue"
    )
    axes[0].legend()

if show_optimal:
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel(r"$\epsilon$   $\alpha$   $c$   $Q_0$")
    axes[1].set_ylabel(f"Proportion optimal action over {avg_over} steps")
    axes[1].plot(epsilons, eps_greedy_optimal, label=r"$\epsilon$-greedy", color="red")
    axes[1].plot(alphas, gradient_optimal, label="gradient bandit", color="green")
    axes[1].plot(confidences, ucb_optimal, label="UCB", color="purple")
    axes[1].plot(Q_0s, optimist_optimal, label=r"greedy with optimistic initialization $\alpha=0.1$", color="black")
    axes[1].plot(
        epsilons, const_step_optimal, label=r"constant step size $\epsilon$-greedy with $\alpha=0.1$", color="blue"
    )
    axes[1].legend()

plt.savefig("media/ex_0211.png")
print("saved figure media/ex_0211.png")
# plt.show()
