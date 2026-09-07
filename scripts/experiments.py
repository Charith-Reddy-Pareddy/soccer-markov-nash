"""Regenerate the experiment CSVs under ``experiments/``.

    python scripts/experiments.py baseline
    python scripts/experiments.py gamma
    python scripts/experiments.py tolerance
    python scripts/experiments.py board          # slow (~10 min)
    python scripts/experiments.py all

Every row comes from ``soccer_nash.experiment.run_config``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.experiment import run_config, standard_goal_rows, write_csv

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments"


def baseline() -> None:
    rows = [
        run_config(move_order=mo, gamma=0.9, measure_exploitability=True)
        for mo in ("deterministic", "coinflip", "random")
    ]
    write_csv(OUT / "baseline.csv", rows)
    print("wrote experiments/baseline.csv")


def gamma_sweep() -> None:
    gammas = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.995]
    rows = []
    for mo in ("deterministic", "random"):
        for g in gammas:
            rows.append(run_config(move_order=mo, gamma=g))
    write_csv(OUT / "gamma_sweep.csv", rows)
    print("wrote experiments/gamma_sweep.csv")


def tolerance_sweep() -> None:
    tols = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
    rows = [
        run_config(move_order="random", gamma=0.9, rel_tol=t) for t in tols
    ]
    write_csv(OUT / "tolerance_sweep.csv", rows)
    print("wrote experiments/tolerance_sweep.csv")


def board_sweep() -> None:
    rows = []
    for width in range(3, 12):
        for height in range(3, 10):
            if width * height > 60:  # keep the state count tractable
                continue
            rows.append(
                run_config(
                    move_order="random",
                    gamma=0.9,
                    width=width,
                    height=height,
                    goal_rows=standard_goal_rows(height),
                )
            )
            print(f"  {width}x{height}: mixed {rows[-1]['mixed_states']}")
    write_csv(OUT / "board_sweep.csv", rows)
    print("wrote experiments/board_sweep.csv")


def goal_mouth_sweep() -> None:
    """Vary the number of goal rows on a fixed 5x9 board."""
    rows = []
    for k in range(1, 8):
        start = (9 - k) // 2
        rows.append(
            run_config(
                move_order="random",
                gamma=0.9,
                width=5,
                height=9,
                goal_rows=tuple(range(start, start + k)),
            )
        )
        print(f"  goal rows = {k}: mixed {rows[-1]['mixed_states']}")
    write_csv(OUT / "goal_mouth_sweep.csv", rows)
    print("wrote experiments/goal_mouth_sweep.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "which",
        choices=["baseline", "gamma", "tolerance", "board", "goalmouth", "all"],
        default="all",
        nargs="?",
    )
    args = parser.parse_args()
    jobs = {
        "baseline": baseline,
        "gamma": gamma_sweep,
        "tolerance": tolerance_sweep,
        "board": board_sweep,
        "goalmouth": goal_mouth_sweep,
    }
    if args.which == "all":
        for fn in jobs.values():
            fn()
    else:
        jobs[args.which]()


if __name__ == "__main__":
    main()
