"""Self-play REINFORCE on the A10 soccer game, checked against the exact
solver -- "does policy gradient converge to a Nash equilibrium" (Sept 17
research meeting), answered empirically over several seeds.

    python scripts/policy_gradient.py --seeds 5   # writes experiments/policy_gradient_seeds.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import A10SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.policy_gradient import evaluate_policy_gradient, train_reinforce_selfplay

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "policy_gradient_seeds.csv"
FIELDS = [
    "seed", "train_time_s", "final_mean_reward",
    "row_action_agreement", "col_action_agreement",
    "mean_equilibrium_regret", "max_equilibrium_regret", "duality_gap",
]


def _fmt(xs: list[float], digits: int = 3) -> str:
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--rollout-len", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    game = A10SoccerGame()

    t = time.perf_counter()
    exact = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10).run()
    t_exact = time.perf_counter() - t
    solver = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10)
    solver.run()
    matrix_of = lambda s: solver._matrix(s, exact.values)  # noqa: E731

    rows: list[dict] = []
    for seed in range(args.seeds):
        t = time.perf_counter()
        result = train_reinforce_selfplay(
            game, gamma=args.gamma, hidden=args.hidden, iterations=args.iterations,
            rollout_len=args.rollout_len, lr=args.lr, seed=seed,
        )
        t_train = time.perf_counter() - t
        m = evaluate_policy_gradient(
            game, result, exact.row_policy, exact.col_policy, matrix_of, gamma=args.gamma,
        )
        rows.append({
            "seed": seed,
            "train_time_s": round(t_train, 1),
            "final_mean_reward": round(result.mean_reward_trace[-1], 4),
            "row_action_agreement": round(m["row_action_agreement"], 4),
            "col_action_agreement": round(m["col_action_agreement"], 4),
            "mean_equilibrium_regret": round(m["mean_equilibrium_regret"], 4),
            "max_equilibrium_regret": round(m["max_equilibrium_regret"], 4),
            "duality_gap": round(m["duality_gap"], 4),
        })
        print(f"  seed {seed}: row agree {rows[-1]['row_action_agreement']:.3f}  "
              f"col agree {rows[-1]['col_action_agreement']:.3f}  "
              f"mean regret {rows[-1]['mean_equilibrium_regret']:.3f}  "
              f"exploit {rows[-1]['duality_gap']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nexact hybrid Nash-Q : {exact.iterations} sweeps, {t_exact:.2f} s")
    print(f"\nself-play REINFORCE ({args.seeds} seeds, {args.iterations} iterations, "
          f"rollout {args.rollout_len}):")
    print(f"  row action agreement   : {_fmt([r['row_action_agreement'] for r in rows])}")
    print(f"  col action agreement   : {_fmt([r['col_action_agreement'] for r in rows])}")
    print(f"  mean equilibrium regret: {_fmt([r['mean_equilibrium_regret'] for r in rows])}")
    print(f"  max equilibrium regret : {_fmt([r['max_equilibrium_regret'] for r in rows])}")
    print(f"  exploitability         : {_fmt([r['duality_gap'] for r in rows])}")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
