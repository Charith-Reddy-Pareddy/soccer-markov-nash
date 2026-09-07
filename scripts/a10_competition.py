"""A10 competition: train the two equilibrium policy networks.

The competition warns that opponents may submit non-equilibrium policies. The
Nash Q policy is the answer: it has a ~0 duality gap (see scripts/selfplay.py),
so no opponent can beat the game value, and it still converts every forced win
an opponent hands it.

This script solves the game, derives each player's security-optimal action set
at every state, fits a bias-free 5->h->h->4 network to each (partial-label
training -- any optimal action counts), reports how exploitable the trained
networks are, and writes the weights in the A10 matrix format.

    python scripts/a10_competition.py --out results/

Writes ``a10_competition.txt`` with "Network First" (player 0) and
"Network Second" (player 1). Every student must train their own.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from soccer_nash.exploit import best_response_to
from soccer_nash.game import SoccerGame
from soccer_nash.mlp import MLP
from soccer_nash.nash_q import NashQIteration


def _train_side(states, X, mask, hidden, epochs, seed, name):
    net = MLP(h1=hidden, h2=hidden, seed=seed)
    net.train(X, mask, epochs=epochs, lr=4e-3, batch_size=256, seed=seed)
    pred = net.predict(X)
    in_opt = float(np.mean(mask[np.arange(len(states)), pred] > 0))
    print(f"  {name}: plays an optimal action at {in_opt:.3f} of states")
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--hidden", type=int, default=99)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--move-order",
        choices=["deterministic", "random", "coinflip"],
        default="deterministic",
    )
    args = parser.parse_args()

    game = SoccerGame(move_order=args.move_order)
    solver = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10)
    nash = solver.run()
    s0 = game.initial_state()
    print(
        f"Nash Q: {nash.iterations} sweeps, V(kickoff) = {nash.values[s0]:+.4f}, "
        f"pure equilibrium: {nash.pure_equilibrium_exists}"
    )

    states, row_mask, col_mask = solver.optimal_action_masks(nash)
    X = np.array(states, dtype=float)

    print("training:")
    net0 = _train_side(states, X, row_mask, args.hidden, args.epochs, args.seed,
                       "Network First (player 0)")
    net1 = _train_side(states, X, col_mask, args.hidden, args.epochs, args.seed + 1,
                       "Network Second (player 1)")

    # How exploitable are the trained networks?
    p0 = net0.policy_dict(states)
    p1 = net1.policy_dict(states)
    from soccer_nash.exploit import onehot_policy

    v1 = best_response_to(game, onehot_policy(p0), responder=1, gamma=args.gamma).values[s0]
    v0 = best_response_to(game, onehot_policy(p1), responder=0, gamma=args.gamma).values[s0]
    print("exploitability (0 = unbeatable beyond the game value):")
    print(f"  Network First : {v1 + nash.values[s0]:.4f}")
    print(f"  Network Second: {v0 - nash.values[s0]:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    text = (
        "# A10 competition networks\n"
        "# Network First (player 0):\n"
        f"{net0.to_a10()}\n"
        "# Network Second (player 1):\n"
        f"{net1.to_a10()}\n"
    )
    (args.out / "a10_competition.txt").write_text(text)
    print(f"wrote {args.out}/a10_competition.txt")


if __name__ == "__main__":
    main()
