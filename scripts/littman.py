"""Reproduce Littman (1994)'s soccer game and its Figure 2 mixed equilibrium.

Littman's grid is 4 rows x 5 columns, two-cell goals in the middle of each end,
five actions (N, S, E, W, stand), moves resolved in a random order, and a 0.1
per-step chance of a declared draw -- i.e. gamma = 0.9. His Figure 2 shows a
state where the ball carrier, pinned near its own goal by the defender, has a
*mixed* optimal policy: randomize between standing and moving away.

This script solves that game exactly (`NashQIteration`, hybrid) with and without
the stand action, locates the Figure-2 state, prints its stage game and the
equilibrium mix, and writes the policy figure to docs/figures/gallery/.

    python scripts/littman.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame, State
from soccer_nash.matrix_games import solve_zero_sum
from soccer_nash.nash_q import NashQIteration
from soccer_nash.templates import describe_action, equilibrium_support
from soccer_nash.viz import policy_svg, strategy_bars_svg

# Littman's board in this repo's (width, height) convention: 5 wide, 4 tall,
# goal on the middle two rows of each end.
BOARD = {"width": 5, "height": 4, "goal_rows": (1, 2), "move_order": "random"}
GAMMA = 0.9
OUT = pathlib.Path("docs/figures/gallery")


def solve(n_actions: int):
    game = SoccerGame(n_actions=n_actions, **BOARD)
    solver = NashQIteration(game, gamma=GAMMA, mode="hybrid", tol=1e-10)
    return game, solver, solver.run()


def _stand_weight(policy: dict[State, np.ndarray], s: State) -> float:
    v = policy[s]
    return float(v[4]) if len(v) == 5 else 0.0


def figure_2_state(game, result) -> State:
    """The Figure-2 state: carrier holds the ball, the defender is on an adjacent
    cell, and the carrier's equilibrium puts real weight on *standing*."""
    best, best_key = None, ()
    for s in result.no_saddle_states:
        x0, y0, x1, y1, b = s
        (cx, cy), (dx, dy) = ((x0, y0), (x1, y1)) if b == 0 else ((x1, y1), (x0, y0))
        adjacent = abs(cx - dx) + abs(cy - dy) == 1
        if not adjacent:
            continue
        carrier_pol = result.row_policy[s] if b == 0 else result.col_policy[s]
        stand = float(carrier_pol[4]) if len(carrier_pol) == 5 else 0.0
        toward_own = cx if b == 0 else game.width - 1 - cx   # distance behind
        key = (stand > 0.05, -abs(stand - 0.5), -toward_own)
        if best is None or key > best_key:
            best, best_key = s, key
    return best


def describe_support(game, s: State, solver, result) -> str:
    m = solver._matrix(s, result.values)
    ri, ci, sub = equilibrium_support(m)
    _, p, q = solve_zero_sum(m)
    b = s[4]
    carrier_w, carrier_idx = (p, ri) if b == 0 else (q, ci)
    defend_w, defend_idx = (q, ci) if b == 0 else (p, ri)
    lines = [f"  state {s}  (player {b} carries)  V = {result.values[s]:+.3f}"]
    lines.append("  carrier support: " + ", ".join(
        f"{describe_action(game, s, b, int(a))} {carrier_w[a] * 100:.0f}%"
        for a in carrier_idx
    ))
    lines.append("  defender support: " + ", ".join(
        f"{describe_action(game, s, 1 - b, int(a))} {defend_w[a] * 100:.0f}%"
        for a in defend_idx
    ))
    lines.append("  stage game (carrier maximises rows):")
    for row in (sub if b == 0 else -sub.T):
        lines.append("    " + "  ".join(f"{v:+.3f}" for v in row))
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    solved = {}
    for n in (4, 5):
        game, solver, result = solve(n)
        solved[n] = (game, solver, result)
        mixed = list(result.no_saddle_states)
        stand_support = sum(
            1 for s in mixed
            if _stand_weight(result.row_policy, s) > 0.02
            or _stand_weight(result.col_policy, s) > 0.02
        )
        rows.append(
            (n, len(solver._states), len(mixed), stand_support,
             result.values[game.initial_state()])
        )

    print(f"Littman soccer  {BOARD['width']}x{BOARD['height']} grid, "
          f"goal rows {BOARD['goal_rows']}, random move order, gamma {GAMMA}\n")
    print(f"{'actions':>8} {'states':>7} {'mixed':>6} {'stand in support':>17} "
          f"{'V(kickoff)':>11}")
    for n, ns, nmix, nstand, v in rows:
        tag = "N,S,E,W" if n == 4 else "N,S,E,W,stand"
        print(f"{tag:>8} {ns:>7} {nmix:>6} {nstand:>17} {v:>+11.3f}")

    four = set(solved[4][2].no_saddle_states)
    five = set(solved[5][2].no_saddle_states)
    print(f"\nno-pure-saddle states: {len(four & five)} shared, "
          f"{len(four - five)} only without stand, {len(five - four)} only with")

    game, solver, result = solved[5]
    s = figure_2_state(game, result)
    print("\nFigure 2 (carrier pinned near its own goal, defender adjacent):")
    print(describe_support(game, s, solver, result))

    fig = policy_svg(game, s, result.row_policy, result.col_policy,
                     result.values[s], title=f"Littman Figure 2  -  state {s}")
    (OUT / "littman_fig2.svg").write_text(fig)
    bars = strategy_bars_svg(result.row_policy[s], result.col_policy[s],
                             result.values[s])
    (OUT / "littman_fig2_bars.svg").write_text(bars)
    print(f"\nwrote {OUT / 'littman_fig2.svg'}")
    print(f"wrote {OUT / 'littman_fig2_bars.svg'}")


if __name__ == "__main__":
    main()
