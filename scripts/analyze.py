"""Summarise Nash Q-iteration over a range of discount factors.

Reports, per gamma: iterations to converge, the fraction of stage games that
admit a pure saddle point, the value of the kickoff state, and the largest
discrepancy between the pure-hybrid and LP-mixed value functions.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration


def run(gammas: list[float], move_order: str) -> None:
    game = SoccerGame(move_order=move_order)
    s0 = game.initial_state()

    print(f"move_order = {move_order}")
    header = f"{'gamma':>6} {'iters':>6} {'saddle%':>9} {'V(kick)':>9} {'|hyb-mix|':>10}"
    print(header)
    print("-" * len(header))

    for gamma in gammas:
        hybrid = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-9).run()
        mixed = NashQIteration(game, gamma=gamma, mode="mixed", tol=1e-9).run()
        gap = max(abs(hybrid.values[s] - mixed.values[s]) for s in hybrid.values)
        print(
            f"{gamma:6.2f} {hybrid.iterations:6d} "
            f"{100 * hybrid.saddle_fraction:8.2f}% "
            f"{hybrid.values[s0]:9.4f} {gap:10.2e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
    )
    parser.add_argument(
        "--move-order",
        choices=["deterministic", "random"],
        default="deterministic",
    )
    args = parser.parse_args()
    run(args.gammas, args.move_order)


if __name__ == "__main__":
    main()
