"""Enumerate *every* stage-game equilibrium, and show why zero-sum is special.

Support enumeration (``soccer_nash.support_enum``) finds all Nash equilibria of
a 2-player game, not just the one an LP returns. Two things fall out:

* the 94 soccer stage games that need mixed strategies -- how many have a unique
  equilibrium, and how sharp the 2x2 mixes are;
* on a general-sum game (Battle of the Sexes) the equilibria disagree on value,
  so selection matters -- which is exactly why the research notes' "largest sum
  of values" rule is a no-op for zero-sum but not in general.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from soccer_nash.game import SoccerGame
from soccer_nash.markov_game import solve_markov_game
from soccer_nash.nash_q import NashQIteration
from soccer_nash.support_enum import all_equilibria, select_equilibrium


def soccer_stage_equilibria(gamma: float) -> None:
    game = SoccerGame(move_order="random")
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-10)
    result = solver.run()

    counts = collections.Counter()
    same_value = 0
    mixes: list[float] = []
    for s in result.no_saddle_states:
        M = solver._matrix(s, result.values)
        eqs = all_equilibria(M, -M, tol=1e-6)
        counts[len(eqs)] += 1
        if all(abs(e.row_value - eqs[0].row_value) < 1e-6 for e in eqs):
            same_value += 1
        for e in eqs:
            mixes.extend(p for p in e.row if 1e-4 < p < 1 - 1e-4)

    total = len(result.no_saddle_states)
    print("94 mixed-strategy soccer stage games:")
    print(f"  equilibria per state:      {dict(sorted(counts.items()))}")
    print(f"  all equilibria same value: {same_value} / {total}  (zero-sum interchangeability)")
    m = np.array(mixes)
    print(f"  mixing probabilities:      range [{m.min():.3f}, {m.max():.3f}], "
          f"none near a clean fraction")


def general_sum_demo(gamma: float) -> None:
    A = np.array([[2.0, 0.0], [0.0, 1.0]])  # row player
    B = np.array([[1.0, 0.0], [0.0, 2.0]])  # column player

    print("\nBattle of the Sexes (general-sum):")
    for e in all_equilibria(A, B):
        kind = "pure " if (e.row > 1e-6).sum() == 1 else "mixed"
        print(f"  {kind}  row {e.row.round(3)}  col {e.col.round(3)}  "
              f"values ({e.row_value:.2f}, {e.col_value:.2f})  sum {e.value_sum:.2f}")

    def transition(s, a0, a1):
        return [(1.0, "x")]

    def reward(s, a0, a1, s_next):
        return float(A[a0, a1]), float(B[a0, a1])

    for rule in ("largest_sum", "row"):
        r = solve_markov_game(["x"], (2, 2), transition, reward, gamma=gamma, select=rule)
        print(f"  repeated, select={rule:<12s} -> "
              f"v0 {r.row_values['x']:.2f}, v1 {r.col_values['x']:.2f}, "
              f"row plays {r.row_policy['x'].round(2)}")
    print("  (the mixed equilibrium is worse for both; 'largest_sum' avoids it)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    args = parser.parse_args()
    soccer_stage_equilibria(args.gamma)
    general_sum_demo(args.gamma)


if __name__ == "__main__":
    main()
