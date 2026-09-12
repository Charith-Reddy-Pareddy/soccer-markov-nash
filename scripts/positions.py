"""Player positions and their Q matrix, in the format sketched at the
research meeting: a board showing where the two players actually are, next to
the full 4x4 Q matrix redrawn as a node-and-arrow graph -- one node per cell
of the *complete* action grid (every case is the full `{U,D,L,R} x {U,D,L,R}`
matrix, never reduced, per the professor's explicit correction), an arrow
toward whichever cell either player would rather deviate to. A pure saddle is
the one node with no outgoing arrow; a matching-pennies game has every node
pointing somewhere, so the arrows chase each other around a closed loop and
no cell is safe -- exactly the picture drawn on the whiteboard.

Twelve cases. Across the whole project there are exactly four *canonical*
equilibrium-support shapes once every stage game is reoriented so rows are
always the carrier's actions (`(2,2)`: 68 states, `(3,3)`: 4, `(2,1)`: 14,
`(3,2)`: 8) -- the defender's support is **never larger** than the
carrier's, in any of the 94 mixed states on the canonical board. This file
shows an example of all four shapes, plus every distinct *mechanism* studied
in the project that can force a mix (a stochastic move order, this project's
own tackle rule, a dense reward with zero transition noise, and movement
slip on an otherwise-always-pure single-cell goal), each answering a
specific question from the meeting:

1. a **pure** state, for contrast -- one node lit up, no cycle, no mixing;
2. the **typical two-action mix** -- a vertical crossing pair, the shape 90 of
   the 94 mixed states actually have (the carrier picking which goal row to
   attack, the defender guessing it);
3. **the corner duel** -- players literally at `(0, 0)` and `(1, 1)`, the
   exact coordinates sketched at the meeting;
4. the **L/R indifference** case -- the meeting's own verbal example ("when
   we move right and when we move left, there's an equal chance of me
   winning"): the carrier's live moves are exactly L and R, no vertical
   option in its support at all, one of only 4 states shaped this way;
5. a genuine **3-action** mix (support `3x3`), requested explicitly ("don't
   focus on entropy, the matrix would be an output by the function");
6. a **near-pure hedge** -- entropy 0.17 bits, the rounding trap made
   concrete: reading 97.5% as 100% would be the mistake the meeting flagged;
7. the typical-mix case's own **twin, mirrored** -- board flipped, value
   negated to machine precision, showing the shape in (2) is not a one-off;
8. this project's own **tackle rule**, a different collision mechanism
   entirely, producing the same kind of duel;
9. the **asymmetric mix** -- support `(2,1)`: the carrier still needs two
   actions while the defender's equilibrium is a single fixed move. Not a
   symmetric duel at all, and the exact "is this really mixed, or just
   another representation of a tie" question from the meeting;
10. the **three-lane mix** -- support `(3,2)`: the carrier genuinely needs
    three actions but the defender only ever needs two to cover them, the
    fourth and last canonical shape, and (by coincidence) the single
    deepest, most rounding-proof gap of any state on this page;
11. **mixing forced by reward alone** -- fully deterministic movement, a
    dense reward that pays for field position instead of goals alone, and
    still a near-fair-coin mix: no stochastic transition anywhere in this
    one;
12. **movement slip on a single-cell goal** -- the one board shape that is
    *always* pure under every move-order rule in this project, forced to mix
    anyway once every player has a chance of slipping to a random move
    regardless of who is where.

    python scripts/positions.py

Writes `docs/figures/gallery/positions.svg` and prints each state's exact
Q matrix and policy.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.certificate import certify_game
from soccer_nash.game import SoccerGame
from soccer_nash.geometry import features
from soccer_nash.nash_q import NashQIteration
from soccer_nash.symmetry import mirror_state
from soccer_nash.viz import bestresponse_graph_svg, panel_svg, policy_svg

FIG = pathlib.Path("docs/figures/gallery/positions.svg")
FIG_WEB_BOARDS = pathlib.Path("docs/figures/gallery/positions_web.svg")
FIG_WEB_MATRIX = pathlib.Path("docs/figures/gallery/positions_web_matrix.svg")
_ACT = ["U", "D", "L", "R"]
CANON = {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "random"}


def _oriented_matrix(solver, state, values):
    """Reorient so rows are always the *carrier's* (maximising) actions and
    columns the defender's, regardless of which player id carries -- matches
    `soccer_nash.numerics._first_2x2_mixed`'s convention, so a reader never
    has to mentally swap row/col meaning between states."""
    M = solver._matrix(state, values)
    return M if state[4] == 0 else -M.T


def _solve(**kw):
    g = SoccerGame(**kw)
    s = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    return g, s, s.run()


def _report(label, g, solver, r, state, panels):
    """Print the full, un-reduced 4x4 Q matrix and policy, and append the
    board + best-response-graph panel pair. The whole 4x4 grid is always
    shown -- no dominance reduction -- so every case is directly comparable
    and matches the professor's own 4x4 format."""
    M = _oriented_matrix(solver, state, r.values)
    cert = certify_game(M)
    print(f"state {state} -- {label}")
    if cert.kind == "pure":
        print(f"  pure equilibrium, saddle at {_ACT[cert.saddle[0]]}/{_ACT[cert.saddle[1]]}")
    else:
        print(f"  mixed equilibrium, no pure saddle, gap {cert.gap:.4f}")
    print("  Q matrix (carrier's payoff; rows = carrier, cols = defender):")
    print("        " + "".join(f"{a:>8}" for a in _ACT))
    for i, a in enumerate(_ACT):
        print(f"    {a:>3} " + "".join(f"{M[i, j]:>8.3f}" for j in range(4)))
    print(f"  row_policy (p0) = {np.round(r.row_policy[state], 3).tolist()}")
    print(f"  col_policy (p1) = {np.round(r.col_policy[state], 3).tolist()}\n")

    panels.append(policy_svg(g, state, r.row_policy, r.col_policy,
                              value=r.values[state], title=label))
    panels.append(bestresponse_graph_svg(M, _ACT, _ACT,
                                          title=f"Q matrix -- {state}"))


def _web_pair(board_panels, matrix_panels, g, solver, r, state, title):
    """A board + Q-matrix-graph pair for the compact website figures (a
    single wide row each, unlike the tall board+graph pairs in FIG)."""
    board_panels.append(policy_svg(g, state, r.row_policy, r.col_policy,
                                    value=r.values[state], title=title))
    M = _oriented_matrix(solver, state, r.values)
    matrix_panels.append(bestresponse_graph_svg(M, _ACT, _ACT,
                                                  title=f"Q matrix {state}"))


def main() -> None:
    g, solver, r = _solve(**CANON)
    mixed = set(r.no_saddle_states)
    panels: list[str] = []
    web_boards: list[str] = []
    web_matrices: list[str] = []

    cases = [
        ((4, 0, 5, 0, 0), "A pure state -- one safe cell, nothing to guess", False),
        ((0, 1, 1, 1, 0), "The typical mix -- which goal row to head for", True),
        ((0, 0, 1, 1, 0), "The corner duel -- (0,0) and (1,1)", True),
        ((1, 1, 1, 0, 1), "A clean L/R indifference example", True),
        ((0, 0, 2, 0, 0), "A genuine 3-action mix, not a 2-cycle", True),
        ((1, 1, 2, 0, 1), "A near-pure hedge -- where rounding would lie", True),
    ]
    # cases featured on the (lighter, five-case) website, keyed by state
    web_titles = {
        (0, 1, 1, 1, 0): "Two-action mix -- which lane to take",
        (0, 0, 1, 1, 0): "The corner duel",
        (0, 0, 2, 0, 0): "Three-action mix",
    }
    for state, why, should_be_mixed in cases:
        assert (state in mixed) == should_be_mixed, f"{state}: unexpected pure/mixed"
        _report(why, g, solver, r, state, panels)
        if state in web_titles:
            _web_pair(web_boards, web_matrices, g, solver, r, state, web_titles[state])

    # 7: the mirror of the typical-mix case -- same shape, opposite corner
    mstate = mirror_state((0, 1, 1, 1, 0), g.width)
    why = "The typical mix, mirrored -- same shape, value negated"
    v1, vm = r.values[(0, 1, 1, 1, 0)], r.values[mstate]
    print(f"  symmetry check: V(case 2) = {v1:+.6f}, V(mirror) = {vm:+.6f}, "
          f"sum = {v1 + vm:+.2e} (should be ~0)\n")
    _report(why, g, solver, r, mstate, panels)

    # 8: this project's own tackle rule -- a different collision mechanism
    gk, sk, rk = _solve(width=5, height=4, goal_rows=(1, 2),
                         move_order="tackle", tackle_prob=0.5)
    tackle_mixed = set(rk.no_saddle_states)
    kstate = next(s for s in tackle_mixed if features(gk, s)["player_dist"] == 1)
    why = "This project's own tackle rule -- a different mechanism, same duel"
    _report(why, gk, sk, rk, kstate, panels)

    # 9: the asymmetric mix -- support (2,1)
    astate = (0, 2, 1, 2, 0)
    why = "The asymmetric mix -- the carrier still guesses, the defender doesn't"
    assert astate in mixed
    _report(why, g, solver, r, astate, panels)
    _web_pair(web_boards, web_matrices, g, solver, r, astate, "Asymmetric mix")

    # 10: the three-lane mix -- support (3,2), the fourth canonical shape,
    # and (by coincidence) the deepest gap of any state on this page
    lstate = (0, 2, 2, 2, 1)
    why = "The three-lane mix -- support (3,2), the deepest gap on this page"
    assert lstate in mixed
    _report(why, g, solver, r, lstate, panels)

    # 11: deterministic + territory -- mixing forced by reward alone
    gt, st_, rt = _solve(width=7, height=5, goal_rows=(1, 2, 3),
                          move_order="deterministic", scoring="territory",
                          territory_reward=0.05)
    tstate = (4, 4, 5, 4, 0)
    why = "Mixing forced by reward alone -- zero transition randomness"
    _report(why, gt, st_, rt, tstate, panels)
    _web_pair(web_boards, web_matrices, gt, st_, rt, tstate,
              "The surprising case -- zero randomness, still mixes")

    # 12: movement slip on a single-cell goal -- the one shape that is
    # always pure under every move-order rule, forced to mix by slip alone
    gs, ss, rs = _solve(width=5, height=5, goal_rows=(2,),
                         move_order="deterministic", slip=0.15)
    slip_mixed = set(rs.no_saddle_states)
    sstate = next(iter(sorted(slip_mixed)[len(slip_mixed) // 2:]))
    why = "Movement slip -- a single-cell goal, always pure until now"
    assert sstate in slip_mixed
    _report(why, gs, ss, rs, sstate, panels)
    print(f"  ({len(slip_mixed)} mixed states appear under slip=0.15 on a "
          f"goal shape that has exactly 0 at slip=0)\n")

    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(panel_svg(panels, cols=2))
    print(f"wrote {FIG}")
    FIG_WEB_BOARDS.write_text(panel_svg(web_boards, cols=len(web_boards)))
    print(f"wrote {FIG_WEB_BOARDS}")
    FIG_WEB_MATRIX.write_text(panel_svg(web_matrices, cols=len(web_matrices)))
    print(f"wrote {FIG_WEB_MATRIX}")


if __name__ == "__main__":
    main()
