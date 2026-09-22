"""Does *how much* the two players' policy networks share weights change
self-play REINFORCE -- the "network architecture choices" question from the
Sept 17 meeting?

Three architectures, same total training budget (iterations x rollout_len),
same seeds:

* **separate** -- two fully independent ``PolicyNet``s
  (`soccer_nash/policy_gradient.py`'s original `train_reinforce_selfplay`).
* **shared** -- one network; player 1's policy is that same network's
  mirror-image induced policy (`soccer_nash/symmetry.py`'s proven
  left-right anti-symmetry, used as a parameterization). One optimizer.
* **partial** -- a shared trunk with a separate linear head per player
  (``SharedTrunkPolicyNet``).

    python scripts/policy_gradient_architectures.py --seeds 5
    # writes experiments/policy_gradient_architectures.csv
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
from soccer_nash.policy_gradient import (
    evaluate_policy_gradient,
    train_reinforce_selfplay,
    train_reinforce_selfplay_shared,
)

OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "experiments"
    / "policy_gradient_architectures.csv"
)
ARCHS = ["separate", "shared", "partial"]
FIELDS = ["seed"] + [f"{a}_{m}" for a in ARCHS for m in ("row_agree", "col_agree", "duality_gap")]


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
        row = {"seed": seed}
        summary_bits = []
        for arch in ARCHS:
            t = time.perf_counter()
            if arch == "separate":
                result = train_reinforce_selfplay(
                    game, gamma=args.gamma, hidden=args.hidden, iterations=args.iterations,
                    rollout_len=args.rollout_len, lr=args.lr, seed=seed,
                )
            else:
                result = train_reinforce_selfplay_shared(
                    game, architecture=arch, gamma=args.gamma, hidden=args.hidden,
                    iterations=args.iterations, rollout_len=args.rollout_len,
                    lr=args.lr, seed=seed,
                )
            dt = time.perf_counter() - t
            m = evaluate_policy_gradient(
                game, result, exact.row_policy, exact.col_policy, matrix_of, gamma=args.gamma,
            )
            row[f"{arch}_row_agree"] = round(m["row_action_agreement"], 4)
            row[f"{arch}_col_agree"] = round(m["col_action_agreement"], 4)
            row[f"{arch}_duality_gap"] = round(m["duality_gap"], 4)
            summary_bits.append(
                f"{arch} agree {row[f'{arch}_row_agree']:.3f} "
                f"exploit {row[f'{arch}_duality_gap']:.3f} ({dt:.0f}s)"
            )
        rows.append(row)
        print(f"  seed {seed}: " + "  |  ".join(summary_bits))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nexact hybrid Nash-Q : {exact.iterations} sweeps, {t_exact:.2f} s")
    print(f"\n{args.seeds} seeds, {args.iterations} iterations x {args.rollout_len} steps each:")
    for arch in ARCHS:
        print(f"  {arch}:")
        print(f"    row agreement : {_fmt([r[f'{arch}_row_agree'] for r in rows])}")
        print(f"    col agreement : {_fmt([r[f'{arch}_col_agree'] for r in rows])}")
        print(f"    exploitability: {_fmt([r[f'{arch}_duality_gap'] for r in rows])}")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
