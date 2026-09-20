"""Reduce the no-pure-saddle states of the random-move-order game to a small
set of geometric templates, and print the 2x2 subgame for each.

    python scripts/templates.py --gamma 0.9
    python scripts/templates.py --n-actions 5   # adds STAND -- see templates.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.symmetry import mirror_state
from soccer_nash.templates import format_templates, mixed_state_templates
from soccer_nash.viz import bestresponse_graph_svg, panel_svg, policy_svg

_ACT4 = ["U", "D", "L", "R"]
_ACT5 = ["U", "D", "L", "R", "STAND"]


def _oriented_matrix(solver, state, values):
    M = solver._matrix(state, values)
    return M if state[4] == 0 else -M.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--n-actions", type=int, default=4, choices=(4, 5))
    args = parser.parse_args()
    act = _ACT4 if args.n_actions == 4 else _ACT5
    fig = pathlib.Path(
        "docs/figures/gallery/templates.svg" if args.n_actions == 4
        else "docs/figures/gallery/templates_n5.svg"
    )

    game = SoccerGame(move_order="random", n_actions=args.n_actions)
    solver = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10)
    result = solver.run()

    mixed = list(result.no_saddle_states)
    mirror_closed = all(
        mirror_state(s, game.width) in set(mixed) for s in mixed
    )
    pairs = len({frozenset((s, mirror_state(s, game.width))) for s in mixed})
    print(f"n_actions={args.n_actions}: {len(mixed)} mixed states, "
          f"mirror-closed={mirror_closed}, {pairs} mirror pairs\n")

    templates = mixed_state_templates(
        game, mixed, lambda s: solver._matrix(s, result.values)
    )
    print(format_templates(templates))

    mp = [t for t in templates if t.mp is not None and t.mp.is_matching_pennies]
    covered = sum(t.count for t in mp)
    print(f"matching-pennies templates: {len(mp)} / {len(templates)}, "
          f"covering {covered} / {len(mixed)} mixed states")
    print(f"\n=> {len(mixed)} no-pure-saddle states reduce to "
          f"{len(templates)} canonical relative configurations under "
          f"mirror/role-swap symmetry and carrier-frame geometry.")

    # one board+matrix panel per template, tiled 4-wide -- "N templates side
    # by side", the representative state for each and its full matrix
    panels: list[str] = []
    for n, t in enumerate(templates, 1):
        rep = t.representative
        M = _oriented_matrix(solver, rep, result.values)
        title = f"[{n}] {t.count} states"
        panels.append(policy_svg(game, rep, result.row_policy, result.col_policy,
                                  value=result.values[rep], title=title))
        panels.append(bestresponse_graph_svg(M, act, act, title=f"{rep}"))
    fig.parent.mkdir(parents=True, exist_ok=True)
    fig.write_text(panel_svg(panels, cols=4))
    print(f"wrote {fig}")


if __name__ == "__main__":
    main()
