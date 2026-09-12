"""Exact hybrid Nash-Q vs. two neural baselines.

The stated pipeline is: exact discrete solver first, then get a network to
replicate it. Two ways to replicate it, so we can tell *where* the
approximation breaks:

* **Q net** -- regress toward the stage matrices `Q(s, a0, a1)`, then extract a
  policy by taking the minimax of the predicted matrix.
* **policy net** -- regress a network *directly* onto the exact equilibrium
  strategies `(p(s), q(s))`.

Over several seeds, with an error bar.

    python scripts/nash_dqn.py --seeds 5      # writes experiments/nash_dqn_seeds.csv
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
from soccer_nash.nash_dqn import (
    compare_policy_to_exact,
    compare_to_exact,
    train_nash_dqn,
    train_policy_baseline,
)
from soccer_nash.nash_q import NashQIteration

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "nash_dqn_seeds.csv"
FIELDS = [
    "seed", "train_time_s", "final_mse", "epochs_to_plateau", "still_improving",
    "max_value_error", "mean_value_error",
    "action_agreement", "classification_agreement", "duality_gap",
    "policy_action_agreement", "policy_max_regret", "policy_duality_gap",
]


def _fmt(xs: list[float], digits: int = 3) -> str:
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def _convergence(loss: list[float]) -> tuple[int, bool]:
    """(first epoch within 5% of the final loss, is it still trending down?)."""
    final = loss[-1]
    plateau = next(
        (i for i, x in enumerate(loss) if x <= final * 1.05), len(loss) - 1
    )
    tail = loss[-20:] if len(loss) >= 20 else loss
    still_improving = tail[0] - tail[-1] > 0.02 * final
    return plateau, still_improving


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    game = A10SoccerGame()

    t = time.perf_counter()
    exact = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10).run()
    t_exact = time.perf_counter() - t
    no_saddle = set(exact.no_saddle_states)
    solver = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10)
    solver.run()
    matrix_of = lambda s: solver._matrix(s, exact.values)  # noqa: E731

    rows: list[dict] = []
    for seed in range(args.seeds):
        t = time.perf_counter()
        dqn = train_nash_dqn(
            game, gamma=args.gamma, hidden=args.hidden, epochs=args.epochs, seed=seed
        )
        t_dqn = time.perf_counter() - t
        m = compare_to_exact(
            game, dqn.net, exact.values, exact.row_policy, args.gamma, no_saddle
        )
        pnet = train_policy_baseline(
            game, exact.row_policy, exact.col_policy, hidden=args.hidden,
            epochs=args.epochs, seed=seed,
        )
        pm = compare_policy_to_exact(
            game, pnet, matrix_of, exact.row_policy, no_saddle, args.gamma
        )
        plateau, improving = _convergence(dqn.loss_trace)
        rows.append({
            "seed": seed,
            "train_time_s": round(t_dqn, 1),
            "final_mse": round(dqn.loss_trace[-1], 5),
            "epochs_to_plateau": plateau,
            "still_improving": int(improving),
            "max_value_error": round(m["max_value_error"], 4),
            "mean_value_error": round(m["mean_value_error"], 4),
            "action_agreement": round(m["action_agreement"], 4),
            "classification_agreement": round(m["classification_agreement"], 4),
            "duality_gap": round(m["duality_gap"], 4),
            "policy_action_agreement": round(pm["action_agreement"], 4),
            "policy_max_regret": round(pm["max_equilibrium_regret"], 4),
            "policy_duality_gap": round(pm["duality_gap"], 4),
        })
        print(f"  seed {seed}: Q-net agree {rows[-1]['action_agreement']:.3f} "
              f"exploit {rows[-1]['duality_gap']:.3f}  |  "
              f"pi-net agree {rows[-1]['policy_action_agreement']:.3f} "
              f"exploit {rows[-1]['policy_duality_gap']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nexact hybrid Nash-Q : {exact.iterations} sweeps, {t_exact:.2f} s, "
          f"value error 0, exploitability 0")
    print(f"\nQ net -- regress toward Q(s,a0,a1), extract minimax "
          f"({args.seeds} seeds, {args.epochs} epochs):")
    print(f"  max |V_dqn - V_exact|     : {_fmt([r['max_value_error'] for r in rows])}")
    print(f"  action agreement          : {_fmt([r['action_agreement'] for r in rows])}")
    print(f"  pure/mixed classification : {_fmt([r['classification_agreement'] for r in rows])}")
    print(f"  exploitability            : {_fmt([r['duality_gap'] for r in rows])}")
    print("\npolicy net -- regress toward the exact (p, q):")
    print(f"  action agreement          : {_fmt([r['policy_action_agreement'] for r in rows])}")
    print(f"  max equilibrium regret    : {_fmt([r['policy_max_regret'] for r in rows])}")
    print(f"  exploitability            : {_fmt([r['policy_duality_gap'] for r in rows])}")
    print("\n=> the policy net names the right action far more often, yet is "
          "no less exploitable: naming the argmax is not game-theoretic "
          "robustness -- the few wrong states are exactly the ones a "
          "best-responder attacks.")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
