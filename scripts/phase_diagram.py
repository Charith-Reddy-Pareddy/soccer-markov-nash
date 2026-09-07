"""Phase diagram: mixed-strategy fraction vs. board geometry and goal-mouth
width, to test whether goal width -- not board size -- is what drives mixing.

Writes ``experiments/phase_diagram.csv``: one row per (width, height, goal
width) on the random-move-order game at a fixed gamma.

    python scripts/phase_diagram.py            # ~5-8 min
    python scripts/phase_diagram.py --quick    # small boards only
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.numerics import classify_stage_game
from soccer_nash.reachability import reachable_states

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "phase_diagram.csv"
FIELDS = [
    "width", "height", "goal_width", "gamma",
    "states", "reachable", "mixed_states", "mixed_fraction",
    "mixed_fraction_reachable", "iterations", "lp_calls", "runtime_s",
]


def centered_goal_rows(height: int, k: int) -> tuple[int, ...]:
    start = (height - k) // 2
    return tuple(range(start, start + k))


def run_one(width: int, height: int, k: int, gamma: float) -> dict:
    goal_rows = centered_goal_rows(height, k)
    game = SoccerGame(width=width, height=height, goal_rows=goal_rows,
                      move_order="random")
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-9)

    t = time.perf_counter()
    result = solver.run()
    runtime = time.perf_counter() - t

    reach = reachable_states(game)
    mixed = 0
    mixed_reach = 0
    for s in solver._states:
        if classify_stage_game(solver._matrix(s, result.values)) == "mixed":
            mixed += 1
            if s in reach:
                mixed_reach += 1
    n = len(solver._states)
    return {
        "width": width,
        "height": height,
        "goal_width": k,
        "gamma": gamma,
        "states": n,
        "reachable": len(reach),
        "mixed_states": mixed,
        "mixed_fraction": round(mixed / n, 6),
        "mixed_fraction_reachable": round(mixed_reach / len(reach), 6),
        "iterations": result.iterations,
        "lp_calls": result.matrix_game_solves,
        "runtime_s": round(runtime, 2),
    }


def configs(quick: bool) -> list[tuple[int, int]]:
    widths = (3, 5, 7) if quick else (3, 5, 7, 9)
    cap = 30 if quick else 45
    out = []
    for w in widths:
        for h in range(3, 8):
            if w * h <= cap:
                out.append((w, h))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    rows = []
    for w, h in configs(args.quick):
        for k in range(1, h - 1):  # goal width 1 .. height-2
            row = run_one(w, h, k, args.gamma)
            rows.append(row)
            print(f"  {w}x{h} goal {k}: mixed {row['mixed_states']:4d}  "
                  f"({row['mixed_fraction']:.3f})  {row['runtime_s']:.1f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
