"""Exact hybrid Nash-Q vs. a neural stage-game approximation.

The professor's stated pipeline is: exact discrete solver first, then get a
network to replicate it. This measures how close the network gets, over several
random seeds so the numbers come with an error bar.

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
from soccer_nash.nash_dqn import compare_to_exact, train_nash_dqn
from soccer_nash.nash_q import NashQIteration

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "nash_dqn_seeds.csv"
FIELDS = [
    "seed", "train_time_s", "final_mse", "epochs_to_plateau", "still_improving",
    "max_value_error", "mean_value_error",
    "action_agreement", "classification_agreement", "duality_gap",
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
        })
        print(f"  seed {seed}: agree {rows[-1]['action_agreement']:.3f}  "
              f"class {rows[-1]['classification_agreement']:.3f}  "
              f"exploit {rows[-1]['duality_gap']:.3f}  {t_dqn:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nexact hybrid Nash-Q : {exact.iterations} sweeps, {t_exact:.2f} s, "
          f"value error 0, exploitability 0")
    print(f"neural Nash-Q ({args.seeds} seeds, {args.epochs} epochs):")
    print(f"  max |V_dqn - V_exact|     : {_fmt([r['max_value_error'] for r in rows])}")
    print(f"  mean |V_dqn - V_exact|    : {_fmt([r['mean_value_error'] for r in rows])}")
    print(f"  action agreement          : {_fmt([r['action_agreement'] for r in rows])}")
    print(f"  pure/mixed classification : {_fmt([r['classification_agreement'] for r in rows])}")
    print(f"  exploitability            : {_fmt([r['duality_gap'] for r in rows])}")
    print(f"  train time (s)            : {_fmt([r['train_time_s'] for r in rows], 0)}")
    print(f"  epochs to MSE plateau     : {_fmt([r['epochs_to_plateau'] for r in rows], 0)}"
          f"  (of {args.epochs}; still improving at the end: "
          f"{sum(r['still_improving'] for r in rows)}/{args.seeds})")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
