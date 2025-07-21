"""
Upper-Confidence-Bound
"""

from src.bandits import Gambler, UCBGambler, run_experiment

if __name__ == "__main__":
    k = 10
    gambler_templates = [
        Gambler(k, epsilon=0.1),
        UCBGambler(k, c=2, epsilon=0.1),
    ]

    run_experiment(
        k=k,
        run_len=1000,
        num_runs=2000,
        gambler_templates=gambler_templates,
        show_stationary=True,
        show_nonstationary=False,
        show_rewards=True,
        show_optimal=True,
    )
