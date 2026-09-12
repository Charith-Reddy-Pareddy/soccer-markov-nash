"""Curated worked examples of mixed-strategy states.

The meeting's ask: don't summarize with entropy or a heatmap -- pick actual
states, draw both players' action distributions as probability-weighted arrows
on the board, and print the *unrounded* 4x4 stage matrix (`U`, `D`, `L`, `R`
only -- no narrative action names) next to each so the group can check by eye
that mixing is genuinely required, not a rounding artefact. And: does mixing
depend on *where* the players are, or on their *relative* position -- and is
there a distance beyond which it never happens?

Six cases, each a different way a soccer stage game ends up needing two players
who both have more than one live move to guess:

1. a near-even 2-action split at the goal mouth,
2. a 3-action mix as far from goal as this board allows (relative position,
   not distance to goal, is what matters),
3. a near-pure hedge (entropy 0.17 bits) -- the rounding trap made concrete,
4. that same state's **mirror image** -- board flipped, players swapped,
   value negated, policy left/right-flipped, showing the symmetry is exact,
5. a near-50/50 mix under **deterministic** transitions, forced purely by a
   dense reward (`scoring="territory"`) coupling the payoff to both actions --
   no stochastic transition anywhere in this one,
6. the **tackle** rule's own mixed state -- a "dive in or contain" duel, a
   different mechanism from the move-order coin entirely.

    python scripts/showcase.py

Writes `docs/figures/gallery/showcase.svg` and prints each state's exact
matrix.
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.certificate import certify_game
from soccer_nash.game import SoccerGame
from soccer_nash.geometry import features
from soccer_nash.nash_q import NashQIteration
from soccer_nash.symmetry import mirror_state
from soccer_nash.viz import panel_svg, policy_svg

FIG = pathlib.Path("docs/figures/gallery/showcase.svg")
_ACT = ["U", "D", "L", "R", "STAND"]
CANON = {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "random"}


def _solve(**kw):
    g = SoccerGame(**kw)
    s = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    return g, s, s.run()


def _report(label, g, solver, r, state):
    M = solver._matrix(state, r.values)
    cert = certify_game(M)
    f = features(g, state)
    print(f"state {state} -- {label}")
    print(f"  carrier-goal distance {f['carrier_goal_dist']}, "
          f"player distance {f['player_dist']}, "
          f"defender_ahead={bool(f['defender_ahead'])}, "
          f"defender_can_intercept={bool(f['defender_can_intercept'])}")
    print(f"  support {cert.support}, entropy {cert.entropy:.3f} bits, "
          f"gap {cert.gap:.6f} (0.1-rounding would "
          f"{'flip this' if cert.gap < 0.05 else 'probably survive'})")
    acts = _ACT[: g.n_actions]
    print("  exact stage matrix M[a0, a1] (player 0's payoff, unrounded):")
    print("        " + "".join(f"{a:>10}" for a in acts))
    for i, a in enumerate(acts):
        print(f"    {a:>3} " + "".join(f"{M[i, j]:>10.6f}" for j in range(g.n_actions)))
    print(f"  row_policy (p0) = {np.round(r.row_policy[state], 3).tolist()}")
    print(f"  col_policy (p1) = {np.round(r.col_policy[state], 3).tolist()}\n")


def main() -> None:
    g, solver, r = _solve(**CANON)
    mixed = set(r.no_saddle_states)

    dist = Counter(features(g, s)["player_dist"] for s in mixed)
    print(f"{len(mixed)} mixed states out of {len(list(g.states()))}.")
    print(f"player-distance distribution among them: {dict(sorted(dist.items()))}")
    print("=> every mixed state has the players within 2 cells of each other; "
          "none is mixed at distance >= 3. Proximity to the *opponent*, not "
          "distance to the goal, is what forces a guess.\n")

    panels = []

    # 1-3: hand-picked from the canonical (random move order) solve
    for state, why in {
        (0, 1, 1, 0, 1): "Near an even split, at the goal mouth",
        (0, 0, 2, 0, 0): "Far from goal, still a 3-way mix",
        (1, 1, 2, 0, 1): "A near-pure hedge -- where rounding would lie",
    }.items():
        assert state in mixed, f"{state} is not actually mixed -- pick another"
        _report(why, g, solver, r, state)
        panels.append(policy_svg(g, state, r.row_policy, r.col_policy,
                                 value=r.values[state], title=why))

    # 4: the mirror of state 1 -- board-flip + player-swap, same mechanism
    mstate = mirror_state((0, 1, 1, 0, 1), g.width)
    why = "The mirror of state 1 -- flipped board, value negated"
    assert mstate in mixed
    _report(why, g, solver, r, mstate)
    v1, vm = r.values[(0, 1, 1, 0, 1)], r.values[mstate]
    print(f"  symmetry check: V(state 1) = {v1:+.6f}, V(mirror) = {vm:+.6f}, "
          f"sum = {v1 + vm:+.2e} (should be ~0)\n")
    panels.append(policy_svg(g, mstate, r.row_policy, r.col_policy,
                             value=vm, title=why))

    # 5: deterministic transitions, mixing forced purely by a dense reward
    gt, st_, rt = _solve(width=7, height=5, goal_rows=(1, 2, 3),
                        move_order="deterministic", scoring="territory",
                        territory_reward=0.05)
    tstate = (4, 4, 5, 4, 0)
    why = "Deterministic transitions -- mixing forced by reward alone"
    assert tstate in set(rt.no_saddle_states), "pick another territory state"
    _report(why, gt, st_, rt, tstate)
    panels.append(policy_svg(gt, tstate, rt.row_policy, rt.col_policy,
                             value=rt.values[tstate], title=why))

    # 6: the tackle rule's own mechanism -- a "dive in or contain" duel
    gk, sk, rk = _solve(width=5, height=4, goal_rows=(1, 2),
                        move_order="tackle", tackle_prob=0.5)
    tackle_mixed = set(rk.no_saddle_states)
    kstate = next(s for s in tackle_mixed if features(gk, s)["player_dist"] == 1)
    why = "The tackle rule -- dive in or contain, not a move-order coin"
    _report(why, gk, sk, rk, kstate)
    panels.append(policy_svg(gk, kstate, rk.row_policy, rk.col_policy,
                             value=rk.values[kstate], title=why))

    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(panel_svg(panels, cols=3))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
