"""Does batching several *independent* rollouts per gradient step reduce
self-play REINFORCE's variance, versus one long correlated trajectory cut
into windows (`scripts/policy_gradient.py`'s existing default)?

Two ways to match the two arms' budgets, since they trade off against each
other and either one alone is a defensible comparison:

* ``--match steps`` (default) -- same total environment-step budget: the
  baseline uses 1 rollout x 2000 iterations x 100 steps = 200,000 steps; the
  batched arm uses 8 independent rollouts x 250 iterations x 100 steps =
  200,000 steps too, so batching gets 8x fewer gradient *updates* for the
  same data. This is the original comparison
  (`experiments/policy_gradient_batch.csv`) -- it found batching collapses
  to a state-independent policy, and the natural question is whether that
  was actually about update *count*, not total data.
* ``--match updates`` -- same iteration (update) count instead: the batched
  arm still runs 2000 iterations, each now with 8x the rollouts, so it sees
  8x the single arm's total environment steps
  (`experiments/policy_gradient_batch_updates.csv`). This isolates whether
  batching needed more *updates*, or just more *data per update*, to avoid
  collapsing.

    python scripts/policy_gradient_batch.py --seeds 5
    python scripts/policy_gradient_batch.py --seeds 5 --match updates
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

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "experiments"
FIELDS = [
    "seed",
    "single_train_time_s", "single_row_agree", "single_duality_gap",
    "batch_train_time_s", "batch_row_agree", "batch_duality_gap",
]


def _fmt(xs: list[float], digits: int = 3) -> str:
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--rollout-len", type=int, default=100)
    parser.add_argument("--single-iterations", type=int, default=2000)
    parser.add_argument("--n-rollouts", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--match", choices=["steps", "updates"], default="steps")
    args = parser.parse_args()

    if args.match == "steps":
        # Same total environment-step budget for both arms.
        batch_iterations = args.single_iterations // args.n_rollouts
        out = OUT_DIR / "policy_gradient_batch.csv"
    else:
        # Same update (iteration) count -- batching sees n_rollouts x the data.
        batch_iterations = args.single_iterations
        out = OUT_DIR / "policy_gradient_batch_updates.csv"

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
        single = train_reinforce_selfplay(
            game, gamma=args.gamma, hidden=args.hidden, iterations=args.single_iterations,
            rollout_len=args.rollout_len, n_rollouts=1, lr=args.lr, seed=seed,
        )
        t_single = time.perf_counter() - t
        m_single = evaluate_policy_gradient(
            game, single, exact.row_policy, exact.col_policy, matrix_of, gamma=args.gamma,
        )

        t = time.perf_counter()
        batch = train_reinforce_selfplay(
            game, gamma=args.gamma, hidden=args.hidden, iterations=batch_iterations,
            rollout_len=args.rollout_len, n_rollouts=args.n_rollouts, lr=args.lr, seed=seed,
        )
        t_batch = time.perf_counter() - t
        m_batch = evaluate_policy_gradient(
            game, batch, exact.row_policy, exact.col_policy, matrix_of, gamma=args.gamma,
        )

        rows.append({
            "seed": seed,
            "single_train_time_s": round(t_single, 1),
            "single_row_agree": round(m_single["row_action_agreement"], 4),
            "single_duality_gap": round(m_single["duality_gap"], 4),
            "batch_train_time_s": round(t_batch, 1),
            "batch_row_agree": round(m_batch["row_action_agreement"], 4),
            "batch_duality_gap": round(m_batch["duality_gap"], 4),
        })
        print(f"  seed {seed}: single agree {rows[-1]['single_row_agree']:.3f} "
              f"exploit {rows[-1]['single_duality_gap']:.3f}  |  "
              f"batch({args.n_rollouts}x) agree {rows[-1]['batch_row_agree']:.3f} "
              f"exploit {rows[-1]['batch_duality_gap']:.3f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    single_steps = args.single_iterations * args.rollout_len
    batch_steps = batch_iterations * args.n_rollouts * args.rollout_len
    print(f"\nexact hybrid Nash-Q : {exact.iterations} sweeps, {t_exact:.2f} s")
    print(f"\nmatch={args.match}, {args.seeds} seeds:")
    print(f"  single trajectory (1 x {args.single_iterations} x {args.rollout_len} "
          f"= {single_steps} env steps):")
    print(f"    row agreement : {_fmt([r['single_row_agree'] for r in rows])}")
    print(f"    exploitability: {_fmt([r['single_duality_gap'] for r in rows])}")
    print(f"  batched independent rollouts "
          f"({args.n_rollouts} x {batch_iterations} x {args.rollout_len} "
          f"= {batch_steps} env steps):")
    print(f"    row agreement : {_fmt([r['batch_row_agree'] for r in rows])}")
    print(f"    exploitability: {_fmt([r['batch_duality_gap'] for r in rows])}")
    print(f"wrote {out.relative_to(out.parent.parent)}")


if __name__ == "__main__":
    main()
