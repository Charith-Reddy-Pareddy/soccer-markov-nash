"""Player positions and their stage game, in the format sketched at the
research meeting: a board showing where the two players actually are, next to
the payoff matrix redrawn as a node-and-arrow graph -- one node per cell, an
arrow toward whichever cell either player would rather deviate to. A pure
saddle is the one node with no outgoing arrow; a matching-pennies game has
every node pointing somewhere, so the arrows chase each other around a closed
loop and no cell is safe -- exactly the picture drawn on the whiteboard.

Three cases, each answering a specific question from the meeting:

1. a **pure** state, for contrast -- one node lit up, no cycle, no mixing;
2. the **L/R indifference** case -- the meeting's own example ("when we move
   right and when we move left, there's an equal chance of me winning"): the
   carrier's live moves are exactly L and R, no vertical option in its support
   at all, so this is a real "the two ways of doing it are equally good" mix,
   not a rock-paper-scissors-style story;
3. a genuine **3-action** mix, requested explicitly ("don't focus on entropy,
   the matrix would be an output by the function").

    python scripts/positions.py

Writes `docs/figures/gallery/positions.svg` and prints each state's matrix.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.certificate import certify_game
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.numerics import essential_subgame
from soccer_nash.viz import bestresponse_graph_svg, panel_svg, policy_svg

FIG = pathlib.Path("docs/figures/gallery/positions.svg")
_ACT = ["U", "D", "L", "R", "STAND"]
CANON = {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "random"}


def _oriented_matrix(solver, state, values):
    """Player 0's payoff matrix, rows = player 0's actions, whichever player
    actually carries the ball (matches soccer_nash.numerics' convention)."""
    M = solver._matrix(state, values)
    return M if state[4] == 0 else -M.T


def main() -> None:
    g = SoccerGame(**CANON)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    r = solver.run()
    mixed = set(r.no_saddle_states)
    acts = _ACT[: g.n_actions]

    cases = [
        ((4, 0, 5, 0, 0), "A pure state -- one safe cell, nothing to guess"),
        ((1, 1, 1, 0, 1), "The meeting's example -- indifferent between L and R"),
        ((0, 0, 2, 0, 0), "A genuine 3-action mix, not a 2-cycle"),
    ]
    panels = []
    for state, why in cases:
        is_mixed = state in mixed
        print(f"state {state} -- {why}")
        print(f"  {'mixed' if is_mixed else 'pure'} equilibrium")
        M = _oriented_matrix(solver, state, r.values)
        cert = certify_game(M)
        if cert.kind == "pure":
            print(f"  saddle at {acts[cert.saddle[0]]}/{acts[cert.saddle[1]]}")
        else:
            print(f"  no pure saddle, gap {cert.gap:.4f}")
        print("  " + "".join(f"{a:>8}" for a in acts))
        for i, a in enumerate(acts):
            print(f"  {a:>3} " + "".join(f"{M[i, j]:>8.3f}" for j in range(g.n_actions)))
        print(f"  row_policy (p0) = {np.round(r.row_policy[state], 3).tolist()}")
        print(f"  col_policy (p1) = {np.round(r.col_policy[state], 3).tolist()}\n")

        panels.append(policy_svg(g, state, r.row_policy, r.col_policy,
                                  value=r.values[state], title=why))
        # reduce to the essential subgame -- the whiteboard drew the 2x2 core,
        # not the full 4x4, and the extra rows/cols are visual noise once
        # they're strictly dominated
        ri, ci, sub = essential_subgame(M)
        row_labels = [acts[i] for i in ri]
        col_labels = [acts[j] for j in ci]
        panels.append(bestresponse_graph_svg(sub, row_labels, col_labels,
                                              title=f"stage game {state}"))

    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(panel_svg(panels, cols=2))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
