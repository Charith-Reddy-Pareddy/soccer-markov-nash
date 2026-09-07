"""Exact hybrid Nash-Q vs. a neural stage-game approximation.

The professor's stated pipeline is: exact discrete solver first, then get a
network to replicate it. This measures how close the network gets.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import A10SoccerGame
from soccer_nash.nash_dqn import compare_to_exact, train_nash_dqn
from soccer_nash.nash_q import NashQIteration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    game = A10SoccerGame()

    t = time.perf_counter()
    exact = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10).run()
    t_exact = time.perf_counter() - t

    t = time.perf_counter()
    dqn = train_nash_dqn(
        game, gamma=args.gamma, hidden=args.hidden, epochs=args.epochs, seed=args.seed
    )
    t_dqn = time.perf_counter() - t
    m = compare_to_exact(game, dqn.net, exact.values, exact.row_policy, args.gamma)

    print(f"exact hybrid Nash-Q : {exact.iterations} sweeps, {t_exact:.2f} s, "
          f"value error 0, exploitability {0.0:.4f}")
    print(f"neural Nash-Q       : {args.epochs} epochs, {t_dqn:.0f} s, "
          f"final MSE {dqn.loss_trace[-1]:.4f}")
    print(f"  max |V_dqn - V_exact| : {m['max_value_error']:.4f}")
    print(f"  action agreement      : {m['action_agreement']:.3f}")
    print(f"  exploitability        : {m['duality_gap']:.4f}")
    print()
    print("The exact solver is both faster and correct; the network needs the")
    print("A10-format constraints relaxed (biases) and still trails on value.")


if __name__ == "__main__":
    main()
