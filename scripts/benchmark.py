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


def _time(make_solver, repeats: int) -> tuple[list[float], object]:
    times, result = [], None
    for _ in range(repeats):
        solver = make_solver()
        t = time.perf_counter()
        result = solver.run()
        times.append(time.perf_counter() - t)
    return times, result


def _row(name: str, times: list[float], result, n_states: int) -> str:
    med = statistics.median(times)
    lp = result.matrix_game_solves
    mixed = len(result.no_saddle_states)
    return (
        f"{name:<22s} {result.iterations:4d} sweeps   "
        f"median {med:6.2f} s  (min {min(times):.2f}, max {max(times):.2f})   "
        f"LP calls {lp:>7d}   "
        f"LP/state/sweep {lp / (n_states * result.iterations):.4f}   "
        f"states needing LP {mixed / n_states:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--full", action="store_true", help="also time all-LP mode")
    args = parser.parse_args()

    game = SoccerGame(move_order="random")
    n = sum(1 for _ in game.states())
    print(f"random move order, 7x5, gamma {args.gamma}, {n} states, "
          f"{args.repeats} repeats\n")

    def hybrid():
        return NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-9)

    times, result = _time(hybrid, args.repeats)
    print(_row("hybrid", times, result, n))

    def sym():
        return NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-9)

    sym_times = []
    sym_result = None
    for _ in range(args.repeats):
        solver = sym()
        t = time.perf_counter()
        sym_result = solver.run_symmetric()
        sym_times.append(time.perf_counter() - t)
    print(_row("hybrid + mirror", sym_times, sym_result, n))

    if args.full:
        def full_lp():
            return NashQIteration(game, gamma=args.gamma, mode="mixed", tol=1e-9)

        lp_times, lp_result = _time(full_lp, 1)
        print(_row("all-LP (1 run)", lp_times, lp_result, n))

    print(
        "\nThe 'states needing LP' fraction is a property of the game and "
        "reproduces exactly;\nthe wall-clock median depends on the machine and "
        "the LP backend."
    )


if __name__ == "__main__":
    main()
