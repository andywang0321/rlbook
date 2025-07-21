"""
Gradient Bandit Algorithm
"""

from src.bandits import GradientGambler, run_experiment

if __name__ == "__main__":
    k = 10
    gambler_templates = [
        GradientGambler(k, alpha=0.1, use_baseline=True),
        GradientGambler(k, alpha=0.1, use_baseline=False),
        GradientGambler(k, alpha=0.4, use_baseline=True),
        GradientGambler(k, alpha=0.4, use_baseline=False),
    ]

    run_experiment(
        k=k,
        run_len=1000,
        num_runs=2000,
        gambler_templates=gambler_templates,
        show_stationary=True,
        show_nonstationary=False,
        show_rewards=False,
        show_optimal=True,
        value=4,
    )
