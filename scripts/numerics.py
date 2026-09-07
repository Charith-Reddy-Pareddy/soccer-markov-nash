"""Numerical behaviour of Nash Q-iteration on the soccer game.

Four questions from the research meeting:

* how to *define* the value of a stage game when the three numbers disagree;
* whether rounding (to force mixed-strategy indifference) is safe under
  discounting;
* which iteration scheme -- value iteration or freeze-then-iterate -- is best;
* where the game actually "has to" mix, and what that sub-game looks like.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.numerics import (
    classify_stage_game,
    is_matching_pennies,
    rounding_changes_saddle,
    support_shape,
)


def value_and_rounding(game, gamma):
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-10)
    r = solver.run()
    gaps = solver.value_bracket_gaps(r)
    cls = collections.Counter()
    flips = 0
    for s in solver._states:
        m = solver._matrix(s, r.values)
        cls[classify_stage_game(m)] += 1
        if rounding_changes_saddle(m, decimals=1):
            flips += 1
    print(f"  stage-game classes:            {dict(cls)}")
    print(f"  value-bracket gap:             max {gaps.max():.1e}, mean {gaps.mean():.1e}")
    print(f"  rounding to 0.1 flips a saddle: {flips} states")
    return solver, r


def iteration_schemes(game, gamma):
    t = time.time()
    vi = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-9).run()
    t_vi = time.time() - t
    t = time.time()
    pi = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-9).run_policy_iteration(
        eval_sweeps=40
    )
    t_pi = time.time() - t
    stale = pi.staleness_trace or []
    lock = next((i for i, x in enumerate(stale) if x < 1e-6), len(stale))
    print(f"  value iteration : {vi.iterations:3d} sweeps, {vi.matrix_game_solves:6d} solves, {t_vi:5.1f} s")
    print(f"  policy iteration: {pi.iterations:3d} rounds, {pi.matrix_game_solves:6d} solves, {t_pi:5.1f} s")
    print(f"  freeze staleness stayed maximal for {lock} rounds, then locked in")


def characterise_mixed_states(solver, result):
    ns = result.no_saddle_states
    if not ns:
        print("  (no states require mixed strategies)")
        return
    shapes = collections.Counter()
    for s in ns:
        m = solver._matrix(s, result.values)
        shapes[support_shape(m)] += 1
    two_by_two = shapes[(2, 2)]
    print(f"  {len(ns)} states need mixed strategies")
    print(f"  equilibrium support shapes:  {dict(sorted(shapes.items()))}")
    print(f"  {two_by_two} of {len(ns)} are 2x2 -- a matching-pennies mix")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--move-orders", nargs="+", default=["deterministic", "random"],
        choices=["deterministic", "random", "coinflip"],
    )
    parser.add_argument("--gamma", type=float, default=0.9)
    args = parser.parse_args()

    for mo in args.move_orders:
        game = SoccerGame(move_order=mo)
        print(f"\n=== move_order = {mo}, gamma = {args.gamma} ===")
        solver, result = value_and_rounding(game, args.gamma)
        iteration_schemes(game, args.gamma)
        characterise_mixed_states(solver, result)


if __name__ == "__main__":
    main()
