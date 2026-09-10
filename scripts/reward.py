"""Three reward objectives for the same soccer dynamics.

The default game scores on a *win*: the first goal ends the game, value =
``P(player 0 wins) - P(player 1 wins)``.

- ``scoring="rate"`` -- a goal scores +1 / -1 and **play continues** from a
  restart, ball to the team that just conceded. Value = expected discounted
  goal difference; a lost position is a floor, not a cliff.
- ``scoring="territory"`` -- ``"win"`` plus a dense per-step reward: holding the
  ball in the opponent's final third earns ``+/- territory_reward`` each step.
  A real objective, not potential-based shaping (this project's own design).

The question is which reward changes leave the no-pure-saddle (mixed) region
where it is. Result:

- **win <-> rate: identical.** The same 94 states, 0 in / 0 out. Changing how
  the *horizon* is scored rescales each stage game ``M(s)`` through the
  continuation value ``V(s')`` but does not cross the ``maximin = minimax``
  boundary.
- **territory: it moves.** A per-step term is added *directly* to ``M(s)``, and
  that can and does cross the boundary -- new mixed stage games appear and some
  leave, more so as ``territory_reward`` grows.

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
_RHO = 0.05  # territory_reward for the headline comparison


def solve(scoring: str, move_order: str, board: dict, gamma: float, rho: float = 0.02):
    game = SoccerGame(move_order=move_order, scoring=scoring,
                      territory_reward=rho, **board)
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
    print(f"{'move order':>13}  {'scoring':>11}  {'mixed':>6}  "
          f"{'V range':>18}  {'V(kickoff)':>11}")

    grid = {}
    combos = [
        ("deterministic", "win", 0.0), ("deterministic", "rate", 0.0),
        ("deterministic", "territory", _RHO),
        ("random", "win", 0.0), ("random", "rate", 0.0),
        ("random", "territory", _RHO),
    ]
    for move_order, scoring, rho in combos:
        game, res = solve(scoring, move_order, board, args.gamma, rho)
        grid[(move_order, scoring)] = (game, res)
        vals = np.fromiter(res.values.values(), dtype=float)
        label = f"territory({rho})" if scoring == "territory" else scoring
        print(f"{move_order:>13}  {label:>11}  {len(res.no_saddle_states):>6}  "
              f"[{vals.min():+.3f}, {vals.max():+.3f}]   "
              f"{res.values[game.initial_state()]:>+11.4f}")

    win_mixed = set(grid[("random", "win")][1].no_saddle_states)
    for sc in ("rate", "territory"):
        m = set(grid[("random", sc)][1].no_saddle_states)
        print(f"\nrandom order, win vs {sc}: win {len(win_mixed)}, {sc} {len(m)}, "
              f"shared {len(win_mixed & m)}, win-only {len(win_mixed - m)}, "
              f"{sc}-only {len(m - win_mixed)}")

    print("\n=> win <-> rate: identical region -- a mixed NE is an equilibrium of "
          "one stage game, and rescaling it through V(s') stays on the same side "
          "of maximin = minimax.")
    print("=> territory: the region moves -- a per-step reward adds a term "
          "directly to M(s), which can cross that boundary.")

    # how far the territory region drifts as the per-step reward grows
    print(f"\n{'territory_reward':>16}  {'mixed':>6}  {'entered':>8}  {'left':>5}  "
          f"{'V(kickoff)':>11}")
    for rho in (0.0, 0.01, 0.02, 0.05, 0.1):
        if rho == 0.0:
            m, vk = win_mixed, grid[("random", "win")][1].values
        else:
            g, r = solve("territory", "random", board, args.gamma, rho)
            m, vk = set(r.no_saddle_states), r.values
        g0 = grid[("random", "win")][0]
        print(f"{rho:>16}  {len(m):>6}  {len(m - win_mixed):>8}  "
              f"{len(win_mixed - m):>5}  {vk[g0.initial_state()]:>+11.4f}")

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

    other = (args.width - 1, args.height // 2)
    g_w, r_w = grid[("random", "win")]
    g_r, r_r = grid[("random", "rate")]
    g_t, r_t = grid[("random", "territory")]
    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(panel_svg([
        value_map_svg(g_w, r_w.values, other, mover=0, ball=0,
                      title="win  (identical mixed region to rate)"),
        value_map_svg(g_r, r_r.values, other, mover=0, ball=0,
                      title="rate  (same 94 states, value compressed)"),
        value_map_svg(g_t, r_t.values, other, mover=0, ball=0,
                      title=f"territory {_RHO}  (region drifts)"),
    ], cols=3))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
