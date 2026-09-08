"""Self-play with the Nash Q policy: empirical outcomes and exploitability.

For each move order:

* roll the Nash policy out against itself from the kickoff and check the
  empirical discounted return against the computed value;
* best-respond to a few player-0 policies and report their exploitability
  (how much the equilibrium value an opponent can beat by best-responding).
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from soccer_nash.best_response import BestResponse
from soccer_nash.exploit import best_response_to, onehot_policy, uniform_policy
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.opponents import part2_opponent
from soccer_nash.simulate import win_rates


def run(move_orders: list[str], gamma: float, n_games: int, seeds: int) -> None:
    for mo in move_orders:
        # the A10 ID-specific kickoff, so all three move orders compare at the
        # same starting state
        game = SoccerGame(move_order=mo, p0_start=(0, 1), p1_start=(6, 3))
        s0 = game.initial_state()
        nq = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-10).run()
        v_nash = nq.values[s0]

        print(f"\n=== move_order = {mo} ===")
        print(f"computed V(kickoff) = {v_nash:+.4f}")

        returns = []
        wr = {}
        for seed in range(seeds):
            wr = win_rates(
                game, nq.row_policy, nq.col_policy, n_games=n_games,
                rng=np.random.default_rng(seed), start=s0, gamma=gamma,
            )
            returns.append(wr["discounted_return"])
        mean = statistics.mean(returns)
        std = statistics.stdev(returns) if seeds > 1 else 0.0
        print(
            f"Nash vs Nash ({n_games} games x {seeds} seeds): "
            f"win {wr['row_win']:.3f}  tie {wr['tie']:.3f}  loss {wr['row_loss']:.3f}"
            f"   discounted return {mean:+.4f} +/- {std:.4f}"
        )

        candidates = {
            "Nash policy": nq.row_policy,
            "uniform random": uniform_policy(game),
        }
        if mo == "deterministic":
            br = BestResponse(game, part2_opponent, me=0, gamma=gamma).solve()
            candidates["best response to scripted (Part 2)"] = onehot_policy(br.policy)

        print("exploitability of player 0 policies:")
        for name, policy in candidates.items():
            v1 = best_response_to(game, policy, responder=1, gamma=gamma).values[s0]
            print(f"  {name:<34s} {v1 + v_nash:6.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--move-orders",
        nargs="+",
        default=["deterministic", "random"],
        choices=["deterministic", "random", "coinflip"],
    )
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--games", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    run(args.move_orders, args.gamma, args.games, args.seeds)


if __name__ == "__main__":
    main()
