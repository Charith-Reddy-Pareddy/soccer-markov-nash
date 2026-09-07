"""Repeated-run benchmark of the pure-first hybrid solver.

Wall-clock is noisy and machine-dependent; the LP-call rate is not. This reports
both, with the runtime as a median over several repeats.

    python scripts/benchmark.py --repeats 5
    python scripts/benchmark.py --repeats 5 --full   # also times all-LP (~8 min)
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration


def _bench(label: str, game: SoccerGame, mode: str, method: str,
           repeats: int, n: int) -> None:
    times, result = [], None
    for _ in range(repeats):
        solver = NashQIteration(game, gamma=0.9, mode=mode, tol=1e-9)
        t = time.perf_counter()
        result = getattr(solver, method)()
        times.append(time.perf_counter() - t)
    med = statistics.median(times)
    lp = result.matrix_game_solves
    mixed = len(result.no_saddle_states)
    print(
        f"{label:<26s} {result.iterations:4d} sweeps   "
        f"median {med:6.2f} s (min {min(times):.2f}, max {max(times):.2f})   "
        f"LP calls {lp:>7d}   "
        f"LP/state/sweep {lp / (n * result.iterations):.4f}   "
        f"states needing LP {mixed / n:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--full", action="store_true", help="also time all-LP mode")
    args = parser.parse_args()

    rnd = SoccerGame(move_order="random")
    det = SoccerGame(move_order="deterministic")
    n = sum(1 for _ in rnd.states())
    print(f"7x5, gamma 0.9, {n} states, {args.repeats} repeats\n")

    _bench("random: hybrid", rnd, "hybrid", "run", args.repeats, n)
    if args.full:
        _bench("random: all-LP", rnd, "mixed", "run", 1, n)
    _bench("deterministic: hybrid", det, "hybrid", "run", args.repeats, n)
    _bench("deterministic: + mirror", det, "hybrid", "run_symmetric", args.repeats, n)

    print(
        "\nThe 'states needing LP' fraction is a property of the game and "
        "reproduces exactly;\nthe wall-clock median depends on the machine and "
        "the LP backend."
    )


if __name__ == "__main__":
    main()
