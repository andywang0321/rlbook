"""
Optimistic initial values
"""

from src.bandits import Gambler, run_experiment

if __name__ == "__main__":
    k = 10
    gambler_templates = [
        Gambler(k, epsilon=0.0, alpha=0.1, initial_values=[5.0] * k),
        Gambler(k, epsilon=0.1, alpha=0.1),
    ]

    run_experiment(
        k=k,
        run_len=100,
        num_runs=2000,
        gambler_templates=gambler_templates,
        labels=[r"$\epsilon=0,\ \alpha=0.1,\ Q_1=5$", r"$\epsilon=0.1,\ \alpha=0.1$"],
        show_stationary=True,
        show_nonstationary=False,
        show_rewards=False,
        show_optimal=True,
    )
