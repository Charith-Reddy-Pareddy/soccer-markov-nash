"""A10 Part 2: train a policy network against the scripted opponent.

Solves the exact best response to the Part 2 opponent (value iteration),
imitates it with a bias-free 5->h->h->4 ReLU/softmax network, then rolls the
network out against the opponent from the kickoff to produce the Q8 trajectory.

    python scripts/a10_part2.py --out results/
    python scripts/a10_part2.py --seeds 5      # imitation accuracy over 5 seeds

Writes ``a10_q7_weights.txt`` (Q7) and ``a10_q8_trajectory.txt`` (Q8).
Every student must train their own network; these files are git-ignored.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from soccer_nash.best_response import BestResponse
from soccer_nash.game import A10SoccerGame
from soccer_nash.mlp import MLP
from soccer_nash.opponents import part2_opponent
from soccer_nash.simulate import play_deterministic


def optimal_action_mask(game, values, gamma):
    """(N, 4) 0/1 mask marking every optimal action at each state."""
    states = list(game.states())
    mask = np.zeros((len(states), 4))
    for i, s in enumerate(states):
        a_opp = part2_opponent(game, s, 1)
        best, acc = -1e18, []
        for a in range(4):
            ns, (r0, _), _ = game.step(s, a, a_opp)
            q = r0 + (0.0 if game.is_terminal(ns) else gamma * values[ns])
            if q > best + 1e-9:
                best, acc = q, [a]
            elif q > best - 1e-9:
                acc.append(a)
        mask[i, acc] = 1.0
    return mask


def _train(game, X, mask, states, hidden, epochs, seed):
    t = time.perf_counter()
    net = MLP(h1=hidden, h2=hidden, seed=seed)
    net.train(X, mask, epochs=epochs, lr=4e-3, batch_size=256, seed=seed)
    dt = time.perf_counter() - t
    pred = net.predict(X)
    in_opt = float(np.mean(mask[np.arange(len(states)), pred] > 0))
    rollout = play_deterministic(game, net.policy_dict(states), part2_opponent, me=0)
    wins = rollout.winner == 0 and rollout.steps <= game.max_steps
    return net, in_opt, dt, wins, rollout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--hidden", type=int, default=99)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=0,
                        help="if >0, train this many seeds and report mean +/- sd")
    args = parser.parse_args()

    game = A10SoccerGame()
    states = list(game.states())

    br = BestResponse(game, part2_opponent, me=0, gamma=args.gamma).solve()
    v0 = br.values[game.initial_state()]
    print(f"best response: {br.iterations} iters, V(kickoff) = {v0:.4f}")

    X = np.array(states, dtype=float)
    mask = optimal_action_mask(game, br.values, args.gamma)

    if args.seeds > 0:
        accs, times, wins = [], [], 0
        for seed in range(args.seeds):
            _, acc, dt, won, _ = _train(game, X, mask, states, args.hidden,
                                        args.epochs, seed)
            accs.append(acc)
            times.append(dt)
            wins += won
            print(f"  seed {seed}: imitation accuracy {acc:.3f}, {dt:.0f}s, "
                  f"wins Q8: {won}")
        sd = statistics.stdev if args.seeds > 1 else (lambda _x: 0.0)
        print(f"\nimitation accuracy : {statistics.mean(accs):.3f} +/- {sd(accs):.3f}")
        print(f"train time (s)     : {statistics.mean(times):.0f} +/- {sd(times):.0f}")
        print(f"wins the Q8 rollout: {wins}/{args.seeds} seeds")
        return

    net, in_opt, _dt, won, rollout = _train(
        game, X, mask, states, args.hidden, args.epochs, args.seed
    )
    print(f"imitation: network plays an optimal action at {in_opt:.3f} of states")
    print(f"Q8 rollout: winner {rollout.winner}, {rollout.steps} steps")
    if not won:
        raise SystemExit("network does not win from the kickoff -- retrain")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "a10_q7_weights.txt").write_text(net.to_a10() + "\n")
    (args.out / "a10_q8_trajectory.txt").write_text(
        "\n".join(",".join(map(str, s)) for s in rollout.trajectory) + "\n"
    )
    print(f"wrote {args.out}/a10_q7_weights.txt and a10_q8_trajectory.txt")


if __name__ == "__main__":
    main()
