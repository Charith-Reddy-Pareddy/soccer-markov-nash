"""Do the players actually visit the mixed-strategy states?

"3.95% of states need the LP" counts the whole state space. This script computes
the discounted state-visitation distribution under the Nash policy from the
kickoff (`soccer_nash/occupancy.py` -- an exact linear fixpoint, not sampling)
and asks how much of the *equilibrium path* runs through the no-pure-saddle
region.

    python scripts/occupancy.py              # 7x5, 3-cell goal, random order

Writes `experiments/occupancy.csv` and the occupancy figure.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.occupancy import concentration, visitation
from soccer_nash.viz import occupancy_map_svg, panel_svg

CSV = pathlib.Path("experiments/occupancy.csv")
FIG = pathlib.Path("docs/figures/gallery/occupancy.svg")


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
    result = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10).run()
    dist = visitation(game, result.row_policy, result.col_policy, gamma=args.gamma)

    all_states = list(game.states())
    mixed = set(result.no_saddle_states)
    mass, uniform = concentration(dist, mixed)
    raw = len(mixed) / len(all_states)

    print(f"{game.width}x{game.height} goal {goal}, random order, gamma {args.gamma}\n")
    print(f"  states reachable under the Nash policy: {len(dist)} / {len(all_states)}")
    print(f"  no-pure-saddle states: {len(mixed)}  ({raw * 100:.2f}% of all)")
    print(f"  equilibrium time spent in mixed states: {mass * 100:.1f}%")
    print(f"  vs. their share of visited states:      {uniform * 100:.1f}%")
    print(f"  => {mass / uniform:.1f}x over-represented on the path, "
          f"{mass / raw:.0f}x vs. the raw state count")

    top = sorted(dist.items(), key=lambda kv: -kv[1])[:8]
    print("\n  most-visited states:")
    for s, m in top:
        print(f"    {s}  {m * 100:5.2f}%   {'mixed' if s in mixed else 'pure'}")

    CSV.parent.mkdir(exist_ok=True)
    with CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x0", "y0", "x1", "y1", "b", "occupancy", "mixed"])
        for s, m in sorted(dist.items(), key=lambda kv: -kv[1]):
            w.writerow([*s, f"{m:.6f}", int(s in mixed)])
    print(f"\nwrote {CSV}")

    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(panel_svg([
        occupancy_map_svg(game, dist, mixed, ball=0,
                          title="player 0 carrying"),
        occupancy_map_svg(game, dist, mixed, ball=1,
                          title="player 1 carrying"),
    ], cols=2))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
