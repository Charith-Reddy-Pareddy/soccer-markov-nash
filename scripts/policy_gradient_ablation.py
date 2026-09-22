"""Two separate REINFORCE variance/stability levers, tested independently
instead of bundled -- a learned value baseline (pure variance reduction:
``advantage = G - V(s)``, does not change the expected gradient) and an
entropy bonus (directly rewards a less-peaked policy, targets premature
collapse, not variance). Four arms per seed: neither, baseline only,
entropy only, both -- so an improvement can be attributed to one mechanism,
the other, or their combination, rather than left ambiguous.

    python scripts/policy_gradient_ablation.py --seeds 3
    # writes experiments/policy_gradient_ablation.csv
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

OUT = (
    pathlib.Path(__file__).resolve().parent.parent / "experiments" / "policy_gradient_ablation.csv"
)
ARMS = [
    ("neither", False, 0.0),
    ("baseline", True, 0.0),
    ("entropy", False, 0.1),
    ("both", True, 0.1),
]
_METRICS = ("row_agree", "col_agree", "duality_gap")
FIELDS = ["seed"] + [f"{name}_{m}" for name, _, _ in ARMS for m in _METRICS]


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
    parser.add_argument("--entropy-coef", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    arms = [
        (name, use_baseline, args.entropy_coef if coef else 0.0)
        for name, use_baseline, coef in ARMS
    ]

    game = A10SoccerGame()

    t = time.perf_counter()
    exact = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10).run()
    t_exact = time.perf_counter() - t
    solver = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10)
    solver.run()
    matrix_of = lambda s: solver._matrix(s, exact.values)  # noqa: E731

    rows: list[dict] = []
    for seed in range(args.seeds):
        row = {"seed": seed}
        summary_bits = []
        for name, use_baseline, entropy_coef in arms:
            t = time.perf_counter()
            result = train_reinforce_selfplay(
                game, gamma=args.gamma, hidden=args.hidden, iterations=args.iterations,
                rollout_len=args.rollout_len, lr=args.lr, seed=seed,
                use_baseline=use_baseline, entropy_coef=entropy_coef,
            )
            dt = time.perf_counter() - t
            m = evaluate_policy_gradient(
                game, result, exact.row_policy, exact.col_policy, matrix_of, gamma=args.gamma,
            )
            row[f"{name}_row_agree"] = round(m["row_action_agreement"], 4)
            row[f"{name}_col_agree"] = round(m["col_action_agreement"], 4)
            row[f"{name}_duality_gap"] = round(m["duality_gap"], 4)
            summary_bits.append(
                f"{name} agree {row[f'{name}_row_agree']:.3f} "
                f"exploit {row[f'{name}_duality_gap']:.3f} ({dt:.0f}s)"
            )
        rows.append(row)
        print(f"  seed {seed}: " + "  |  ".join(summary_bits))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nexact hybrid Nash-Q : {exact.iterations} sweeps, {t_exact:.2f} s")
    print(f"\n{args.seeds} seeds, {args.iterations} iterations x {args.rollout_len} steps each, "
          f"entropy_coef={args.entropy_coef}:")
    for name, _, _ in arms:
        print(f"  {name}:")
        print(f"    row agreement : {_fmt([r[f'{name}_row_agree'] for r in rows])}")
        print(f"    col agreement : {_fmt([r[f'{name}_col_agree'] for r in rows])}")
        print(f"    exploitability: {_fmt([r[f'{name}_duality_gap'] for r in rows])}")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
