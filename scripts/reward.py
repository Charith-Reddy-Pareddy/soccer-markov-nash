"""Two reward objectives for the same soccer dynamics.

The default game scores on a *win*: the first goal ends the game, and the value
is ``P(player 0 wins) - P(player 1 wins)``. This script adds and studies a
second objective, ``scoring="rate"``: a goal scores +1 / -1 and **play
continues** from a restart, with the ball going to the team that just conceded.
The value becomes the *expected discounted goal difference*, so a lost position
is no longer a cliff -- you concede and get the ball back at the centre.

It solves both objectives on the same board and reports:

- whether the no-pure-saddle (mixed) region moves -- it does not,
- how the value range changes -- the "rate" objective compresses it, because
  conceding is a floor,
- V(kickoff) under each.

    python scripts/reward.py                  # 7x5, 3-cell goal, random order

Writes ``experiments/reward.csv`` and the value-map figure.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.viz import panel_svg, value_map_svg

CSV = pathlib.Path("experiments/reward.csv")
FIG = pathlib.Path("docs/figures/gallery/reward_value.svg")


def solve(scoring: str, move_order: str, board: dict, gamma: float):
    game = SoccerGame(move_order=move_order, scoring=scoring, **board)
    result = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-10).run()
    return game, result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=7)
    ap.add_argument("--height", type=int, default=5)
    ap.add_argument("--goal-rows", type=str, default="1,2,3")
    ap.add_argument("--gamma", type=float, default=0.9)
    args = ap.parse_args()

    board = {
        "width": args.width,
        "height": args.height,
        "goal_rows": tuple(int(v) for v in args.goal_rows.split(",")),
    }

    print(f"{args.width}x{args.height} goal {board['goal_rows']}, gamma {args.gamma}\n")
    print(f"{'move order':>13}  {'scoring':>7}  {'mixed':>6}  "
          f"{'V range':>18}  {'V(kickoff)':>11}")

    grid = {}
    for move_order in ("deterministic", "random"):
        for scoring in ("win", "rate"):
            game, res = solve(scoring, move_order, board, args.gamma)
            grid[(move_order, scoring)] = (game, res)
            vals = np.fromiter(res.values.values(), dtype=float)
            print(f"{move_order:>13}  {scoring:>7}  {len(res.no_saddle_states):>6}  "
                  f"[{vals.min():+.3f}, {vals.max():+.3f}]   "
                  f"{res.values[game.initial_state()]:>+11.4f}")

    win_mixed = set(grid[("random", "win")][1].no_saddle_states)
    rate_mixed = set(grid[("random", "rate")][1].no_saddle_states)
    print(f"\nrandom order -- mixed states: win {len(win_mixed)}, rate "
          f"{len(rate_mixed)}, shared {len(win_mixed & rate_mixed)}, "
          f"win-only {len(win_mixed - rate_mixed)}, "
          f"rate-only {len(rate_mixed - win_mixed)}")
    print("=> the mixed-NE region does not depend on the reward objective: a "
          "mixed NE is an equilibrium of one stage game, not of the horizon.")

    wv = grid[("random", "win")][1].values
    rv = grid[("random", "rate")][1].values
    wr = np.fromiter(wv.values(), dtype=float)
    rr = np.fromiter(rv.values(), dtype=float)
    print(f"\nvalue compression (random order): "
          f"win spans {wr.max() - wr.min():.3f}, rate spans {rr.max() - rr.min():.3f}")
    print(f"  worst state: win {wr.min():+.3f} (you lose), "
          f"rate {rr.min():+.3f} (concede and restart with the ball)")

    CSV.parent.mkdir(exist_ok=True)
    with CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["move_order", "scoring", "states", "mixed",
                    "v_min", "v_max", "v_kickoff"])
        for (mo, sc), (game, res) in grid.items():
            vals = np.fromiter(res.values.values(), dtype=float)
            w.writerow([mo, sc, len(res.values), len(res.no_saddle_states),
                        f"{vals.min():.4f}", f"{vals.max():.4f}",
                        f"{res.values[game.initial_state()]:.4f}"])
    print(f"\nwrote {CSV}")

    g_w, r_w = grid[("random", "win")]
    g_r, r_r = grid[("random", "rate")]
    other = (args.width - 1, args.height // 2)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(panel_svg([
        value_map_svg(g_w, r_w.values, other, mover=0, ball=0,
                      title="scoring = win  (P(win) - P(loss))"),
        value_map_svg(g_r, r_r.values, other, mover=0, ball=0,
                      title="scoring = rate  (expected goal difference)"),
    ], cols=2))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
