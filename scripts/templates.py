"""Reduce the no-pure-saddle states of the random-move-order game to a small
set of geometric templates, and print the 2x2 subgame for each.

    python scripts/templates.py --gamma 0.9
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    args = parser.parse_args()

    game = SoccerGame(move_order="random")
    solver = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10)
    result = solver.run()

    mixed = list(result.no_saddle_states)
    mirror_closed = all(
        mirror_state(s, game.width) in set(mixed) for s in mixed
    )
    pairs = len({frozenset((s, mirror_state(s, game.width))) for s in mixed})
    print(f"{len(mixed)} mixed states, mirror-closed={mirror_closed}, "
          f"{pairs} mirror pairs\n")

    templates = mixed_state_templates(
        game, mixed, lambda s: solver._matrix(s, result.values)
    )
    print(format_templates(templates))

    mp = [t for t in templates if t.mp is not None and t.mp.is_matching_pennies]
    covered = sum(t.count for t in mp)
    print(f"matching-pennies templates: {len(mp)} / {len(templates)}, "
          f"covering {covered} / {len(mixed)} mixed states")


if __name__ == "__main__":
    main()
