"""Value iteration vs. freeze-then-iterate policy iteration.

Both reach the same minimax fixed point. Policy iteration solves each stage
game only on its outer rounds, holding the strategies fixed for cheap linear
evaluation sweeps in between -- the "do not recompute Nash every time" idea.
This reports the wall clock, sweep count and matrix-game (LP) solve count for
each.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration


def run(move_orders: list[str], mode: str, gamma: float, eval_sweeps: int) -> None:
    for mo in move_orders:
        game = SoccerGame(move_order=mo)
        s0 = game.initial_state()
        print(f"\n=== move_order = {mo}, stage solver = {mode} ===")

        t = time.time()
        vi = NashQIteration(game, gamma=gamma, mode=mode, tol=1e-9).run()
        t_vi = time.time() - t

        t = time.time()
        pi = NashQIteration(game, gamma=gamma, mode=mode, tol=1e-9).run_policy_iteration(
            eval_sweeps=eval_sweeps
        )
        t_pi = time.time() - t

        gap = max(abs(vi.values[s] - pi.values[s]) for s in vi.values)
        print(f"  V(kickoff): VI {vi.values[s0]:+.5f}   PI {pi.values[s0]:+.5f}   |diff| {gap:.1e}")
        print(
            f"  value iteration : {vi.iterations:4d} sweeps, "
            f"{vi.matrix_game_solves:7d} matrix-game solves, {t_vi:6.1f} s"
        )
        print(
            f"  policy iteration: {pi.iterations:4d} rounds, "
            f"{pi.matrix_game_solves:7d} matrix-game solves, {t_pi:6.1f} s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--move-orders", nargs="+", default=["deterministic", "random"],
        choices=["deterministic", "random", "coinflip"],
    )
    parser.add_argument("--mode", default="hybrid", choices=["pure", "mixed", "hybrid"])
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--eval-sweeps", type=int, default=50)
    args = parser.parse_args()
    run(args.move_orders, args.mode, args.gamma, args.eval_sweeps)


if __name__ == "__main__":
    main()
