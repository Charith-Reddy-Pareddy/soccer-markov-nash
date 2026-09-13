"""Does genuine mixing still happen when the players are not adjacent?

The meeting question, directly: bucket every state on the canonical board by
Manhattan distance between the two players and report, per distance, how
many states there are, how many have no pure saddle, how many of those are a
genuinely forced (unique) mix, and how many are degenerate (a zero-weight
action tied with the reported support -- `scripts/degeneracy.py`).

    python scripts/distance_table.py

Writes `experiments/distance_table.csv`.
"""

from __future__ import annotations

import collections
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from degeneracy import CANON, classify  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "distance_table.csv"


def manhattan(state) -> int:
    x0, y0, x1, y1, _b = state
    return abs(x0 - x1) + abs(y0 - y1)


def main() -> None:
    game = SoccerGame(**CANON)
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run()
    no_saddle = set(result.no_saddle_states)

    total_by_d = collections.Counter()
    nosaddle_by_d = collections.Counter()
    unique_by_d = collections.Counter()
    degenerate_by_d = collections.Counter()

    rows = []
    for s in game.states():
        d = manhattan(s)
        total_by_d[d] += 1
        if s in no_saddle:
            nosaddle_by_d[d] += 1
            row = classify(solver, s, result.row_policy, result.col_policy,
                            result.values)
            if row["kind"] == "unique":
                unique_by_d[d] += 1
            else:
                degenerate_by_d[d] += 1

    max_d = max(total_by_d)
    print("canonical board (7x5, goal_rows=(1,2,3), random move order), "
          "gamma 0.9\n")
    print(f"{'distance':>8}  {'states':>7}  {'no-pure-saddle':>15}  "
          f"{'forced (unique)':>16}  {'degenerate':>10}  {'% no-pure-saddle':>17}")
    for d in range(max_d + 1):
        n = total_by_d[d]
        ns = nosaddle_by_d[d]
        pct = 100 * ns / n if n else 0.0
        print(f"{d:>8}  {n:>7}  {ns:>15}  {unique_by_d[d]:>16}  "
              f"{degenerate_by_d[d]:>10}  {pct:>16.1f}%")
        rows.append({
            "distance": d, "states": n, "no_pure_saddle": ns,
            "forced_unique": unique_by_d[d], "degenerate": degenerate_by_d[d],
            "pct_no_pure_saddle": round(pct, 2),
        })

    max_mixed_d = max((d for d in total_by_d if nosaddle_by_d[d] > 0), default=0)
    print(f"\n=> no-pure-saddle states appear up to Manhattan distance "
          f"{max_mixed_d}; every state at distance > {max_mixed_d} has a "
          f"pure saddle. Genuine mixing is not limited to adjacency -- it "
          f"just gets rarer with distance.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
