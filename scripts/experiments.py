"""Regenerate the experiment CSVs under ``experiments/``.

    python scripts/experiments.py baseline
    python scripts/experiments.py undiscounted
    python scripts/experiments.py gamma
    python scripts/experiments.py tolerance
    python scripts/experiments.py board          # slow (~10 min)
    python scripts/experiments.py all

Every row comes from ``soccer_nash.experiment.run_config``.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.experiment import run_config, standard_goal_rows, write_csv
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments"

# the A10 page's ID-specific kickoff on the 7x5 board (kickoff_value / duality_gap
# are reported at this state; the pure/mixed split does not depend on it)
A10 = {"p0_start": (0, 1), "p1_start": (6, 3)}


def baseline() -> None:
    rows = [
        run_config(move_order=mo, gamma=0.9, measure_exploitability=True, **A10)
        for mo in ("deterministic", "coinflip", "random")
    ]
    write_csv(OUT / "baseline.csv", rows)
    print("wrote experiments/baseline.csv")


def gamma_sweep() -> None:
    gammas = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.995]
    rows = []
    for mo in ("deterministic", "random"):
        for g in gammas:
            rows.append(run_config(move_order=mo, gamma=g, **A10))
    write_csv(OUT / "gamma_sweep.csv", rows)
    print("wrote experiments/gamma_sweep.csv")


def undiscounted() -> None:
    """The exact undiscounted game (backward induction, 100-step horizon)."""
    rows = []
    for mo in ("deterministic", "coinflip", "random"):
        game = SoccerGame(move_order=mo, **A10)
        r = NashQIteration(game, mode="hybrid").run_finite_horizon()
        rows.append({
            "move_order": mo,
            "horizon": r.horizon,
            "states": sum(1 for _ in game.states()),
            "states_mixed_at_some_step": len(r.no_saddle_states),
            "mixed_stage_games": r.mixed_stage_games,
            "kickoff_value": round(r.values[game.initial_state()], 6),
            "pure_equilibrium": r.pure_equilibrium_exists,
        })
        print(f"  {mo}: {rows[-1]}")
    with (OUT / "undiscounted.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote experiments/undiscounted.csv")


def tolerance_sweep() -> None:
    tols = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
    rows = [
        run_config(move_order="random", gamma=0.9, rel_tol=t, **A10) for t in tols
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


def one_cell_goal_sweep() -> None:
    """A single-cell goal across many boards and discounts -- the conjecture is
    that all of these have zero mixed states."""
    rows = []
    configs = [(w, h) for w in (3, 5, 7, 9, 11) for h in (3, 5, 7, 9) if w * h <= 55]
    for w, h in configs:
        rows.append(run_config(
            move_order="random", gamma=0.9, width=w, height=h, goal_rows=(h // 2,),
        ))
        print(f"  {w}x{h}: mixed {rows[-1]['mixed_states']}")
    for gm in (0.5, 0.7, 0.95, 0.99):
        rows.append(run_config(
            move_order="random", gamma=gm, width=7, height=5, goal_rows=(2,),
        ))
        print(f"  7x5 gamma={gm}: mixed {rows[-1]['mixed_states']}")
    write_csv(OUT / "one_cell_goal.csv", rows)
    print("wrote experiments/one_cell_goal.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "which",
        choices=["baseline", "undiscounted", "gamma", "tolerance", "board",
                 "goalmouth", "onecell", "all"],
        default="all",
        nargs="?",
    )
    args = parser.parse_args()
    jobs = {
        "baseline": baseline,
        "undiscounted": undiscounted,
        "gamma": gamma_sweep,
        "tolerance": tolerance_sweep,
        "board": board_sweep,
        "goalmouth": goal_mouth_sweep,
        "onecell": one_cell_goal_sweep,
    }
    if args.which == "all":
        for fn in jobs.values():
            fn()
    else:
        jobs[args.which]()


if __name__ == "__main__":
    main()
