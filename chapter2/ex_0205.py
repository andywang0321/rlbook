"""
Exercise 2.5
Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary problems. Use a modified version of the 10-armed testbed in which all the $q_*(a)$ start out equal and then take independent random walks (say by adding a normally distributed increment with mean zero and standard deviation 0.01 to all the $q_*(a)$ on each step). Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally computed, and another action-value method using a constant step-size parameter, $\\alpha = 0.1$. Use $\\epsilon = 0.1$ and longer runs, say of 10,000 steps.
"""

from src.bandits import Gambler, run_experiment

if __name__ == "__main__":
    gambler_templates = [
        Gambler(k=10, epsilon=0.1, alpha=None),  # Sample-average method
        Gambler(k=10, epsilon=0.1, alpha=0.1),  # Constant step-size method
    ]

    run_experiment(
        k=10,
        run_len=10000,
        num_runs=2000,
        gambler_templates=gambler_templates,
        show_stationary=True,
        show_nonstationary=True,
        show_rewards=True,
        show_optimal=True,
    )
