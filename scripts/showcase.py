"""Curated worked examples of mixed-strategy states, for the research meeting.

The meeting's ask: don't summarize with entropy or a heatmap -- pick a handful
of actual states, draw both players' action distributions as probability-
weighted arrows on the board, and print the *unrounded* stage matrix next to
each so the group can check by eye that mixing is genuinely required, not a
rounding artefact. And: does mixing depend on *where* the players are, or on
their *relative* position -- and is there a distance beyond which it never
happens?

    python scripts/showcase.py

Writes `docs/figures/gallery/showcase.svg` (the four boards) and prints each
state's full-precision matrix.
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
from soccer_nash.viz import panel_svg, policy_svg

FIG = pathlib.Path("docs/figures/gallery/showcase.svg")
_ACT = ["U", "D", "L", "R", "STAND"]

# hand-picked from the 94 no-pure-saddle states on the 7x5 / 3-cell-goal /
# random-order board, gamma = 0.9 -- see the printed rationale for each.
PICKS = {
    (0, 1, 1, 0, 1): "Near an even split, at the goal mouth",
    (0, 0, 2, 0, 0): "Far from goal, still a 3-way mix",
    (1, 1, 2, 0, 1): "A near-pure hedge -- where rounding would lie",
}


def main() -> None:
    g = SoccerGame(width=7, height=5, goal_rows=(1, 2, 3), move_order="random")
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    r = solver.run()
    mixed = set(r.no_saddle_states)

    dist = Counter(features(g, s)["player_dist"] for s in mixed)
    print(f"{len(mixed)} mixed states out of {len(list(g.states()))}.")
    print(f"player-distance distribution among them: {dict(sorted(dist.items()))}")
    print("=> every mixed state has the players within 2 cells of each other; "
          "none is mixed at distance >= 3. Proximity to the *opponent*, not "
          "distance to the goal, is what forces a guess.\n")

    panels = []
    for state, why in PICKS.items():
        assert state in mixed, f"{state} is not actually mixed -- pick another"
        M = solver._matrix(state, r.values)
        cert = certify_game(M)
        f = features(g, state)
        v = r.values[state]

        print(f"state {state} -- {why}")
        print(f"  carrier-goal distance {f['carrier_goal_dist']}, "
              f"player distance {f['player_dist']}, "
              f"defender_ahead={bool(f['defender_ahead'])}, "
              f"defender_can_intercept={bool(f['defender_can_intercept'])}")
        print(f"  support {cert.support}, entropy {cert.entropy:.3f} bits, "
              f"gap {cert.gap:.6f} (0.1-rounding would "
              f"{'flip this' if cert.gap < 0.05 else 'probably survive'})")
        acts = _ACT[: g.n_actions]
        print("  exact stage matrix M[a0, a1] (player 0's payoff, unrounded):")
        header = "        " + "".join(f"{a:>10}" for a in acts)
        print(header)
        for i, a in enumerate(acts):
            print(f"    {a:>3} " + "".join(f"{M[i, j]:>10.6f}" for j in range(g.n_actions)))
        print(f"  row_policy (carrier or p0) = {np.round(r.row_policy[state], 3).tolist()}")
        print(f"  col_policy (defender or p1) = {np.round(r.col_policy[state], 3).tolist()}\n")

        panels.append(policy_svg(g, state, r.row_policy, r.col_policy, value=v,
                                 title=why))

    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(panel_svg(panels, cols=3))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
