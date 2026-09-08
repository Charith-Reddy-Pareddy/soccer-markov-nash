"""How mixed are the mixed stage games, and what is mixing worth?

"No pure saddle" is not "50/50". This script solves the random-order game and
reports three things about its no-pure-saddle states:

1. **How close to a saddle they are** -- the ``minimax - maximin`` gap. On the
   default board every one is under 0.07: these are shallow, not wild, mixes.
2. **Their support structure** -- 2x2 vs wider, and how many 2x2 supports are
   *structurally* matching pennies (best replies cross, a robust `>` test that
   does not depend on the LP vertex).
3. **The value of mixing** -- ``V(hybrid) - V(pure maximin)`` at each state and
   at the kickoff. A player forced to play pure strategies gives this up.

    python scripts/mixing.py                 # 7x5, 3-cell goal, gamma 0.9

Writes ``experiments/mixing.csv``.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.matrix_games import pure_bounds
from soccer_nash.nash_q import NashQIteration
from soccer_nash.templates import equilibrium_support, matching_pennies_pattern

CSV = pathlib.Path("experiments/mixing.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=7)
    ap.add_argument("--height", type=int, default=5)
    ap.add_argument("--goal-rows", type=str, default="1,2,3")
    ap.add_argument("--gamma", type=float, default=0.9)
    args = ap.parse_args()

    goal = tuple(int(v) for v in args.goal_rows.split(","))
    game = SoccerGame(width=args.width, height=args.height, goal_rows=goal,
                      move_order="random")
    hybrid = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10)
    hres = hybrid.run()
    pres = NashQIteration(game, gamma=args.gamma, mode="pure", tol=1e-10).run()

    mixed = sorted(hres.no_saddle_states)
    if not mixed:
        print("no mixed stage games on this board")
        return

    rows = []
    for s in mixed:
        m = hybrid._matrix(s, hres.values)
        lo, hi = pure_bounds(m)
        ri, ci, sub = equilibrium_support(m)
        is_mp = (
            sub.shape == (2, 2)
            and matching_pennies_pattern(sub).is_matching_pennies
        )
        rows.append({
            "state": s,
            "gap": float(hi - lo),
            "support": (len(ri), len(ci)),
            "matching_pennies": is_mp,
            "value_of_mixing": float(hres.values[s] - pres.values[s]),
        })

    gap = np.array([r["gap"] for r in rows])
    vom = np.array([r["value_of_mixing"] for r in rows])
    two_by_two = sum(r["support"] == (2, 2) for r in rows)
    mp = sum(r["matching_pennies"] for r in rows)
    s0 = game.initial_state()

    print(f"{game.width}x{game.height} goal {goal}, random order, gamma {args.gamma}")
    print(f"no-pure-saddle states: {len(rows)}\n")
    print("1. distance to a pure saddle (minimax - maximin)")
    print(f"     median {np.median(gap):.3f}   max {gap.max():.3f}   "
          f"all < 0.1: {bool((gap < 0.1).all())}")
    print("\n2. support structure")
    print(f"     2x2 support: {two_by_two} / {len(rows)}")
    print(f"     structurally matching pennies (crossing best replies): {mp}")
    print("\n3. value of mixing  (V hybrid - V pure maximin)")
    print(f"     at the mixed states: mean {vom.mean():+.3f}  "
          f"median {np.median(vom):+.3f}  max {vom.max():.3f}")
    print(f"     at the kickoff {s0}: "
          f"{hres.values[s0]:+.3f} vs {pres.values[s0]:+.3f}  "
          f"(pure play secures {pres.values[s0]:+.3f})")

    CSV.parent.mkdir(exist_ok=True)
    with CSV.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x0", "y0", "x1", "y1", "b", "gap",
                     "row_support", "col_support", "matching_pennies",
                     "value_of_mixing"])
        for r in rows:
            wr.writerow([*r["state"], f"{r['gap']:.4f}",
                         r["support"][0], r["support"][1],
                         int(r["matching_pennies"]),
                         f"{r['value_of_mixing']:.4f}"])
    print(f"\nwrote {CSV}")


if __name__ == "__main__":
    main()
