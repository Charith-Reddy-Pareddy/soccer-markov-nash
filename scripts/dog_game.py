"""Solve the discretised dog / debate game and check it against the closed form.

The only Nash equilibrium parks each player in its own house; the dog then sits
at ``w_a * house_a + w_b * house_b``. This exercises the general-sum solver
(``markov_game.py`` + support enumeration) on a game with a known answer.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from soccer_nash.dog_game import DogGame
from soccer_nash.markov_game import solve_markov_game


def run(size: int, w_a: float, line: bool) -> None:
    game = DogGame(size=size, w_a=w_a, line=line)
    kind = "1D (zero-sum)" if line else "2D (general-sum)"
    print(f"dog game {kind}: {size}x{'1' if line else size} board, w_a = {w_a}")

    t = time.perf_counter()
    r = solve_markov_game(
        game.states(), game.n_actions, game.transition, game.reward,
        gamma=0.9, tol=1e-7, max_iters=500,
    )
    dt = time.perf_counter() - t

    eq = (game.house_a, game.house_b)
    dog = game.dog(*eq)
    closed = game.equilibrium_dog()
    print(f"  {len(game.states())} states, {r.iterations} sweeps, {dt:.1f} s")
    print(f"  equilibrium: player 0 plays {r.row_policy[eq].round(2)}, "
          f"player 1 plays {r.col_policy[eq].round(2)}")
    print(f"  dog at equilibrium: {tuple(round(x, 3) for x in dog)}   "
          f"closed form: {tuple(round(x, 3) for x in closed)}")
    print(f"  states needing support enumeration: {len(r.support_enum_states)} "
          f"/ {len(game.states())}")
    print(f"  states with multiple equilibria:    {len(r.multi_equilibrium_states)}")
    match = np.allclose(dog, closed, atol=1e-6)
    print(f"  {'MATCH' if match else 'MISMATCH'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--w-a", type=float, default=0.5)
    args = parser.parse_args()
    run(size=5, w_a=args.w_a, line=True)
    run(size=args.size, w_a=args.w_a, line=False)
    run(size=args.size, w_a=0.75, line=False)


if __name__ == "__main__":
    main()
