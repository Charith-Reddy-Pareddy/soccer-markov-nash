"""Width/depth ablation on the Q net -- is 45% action agreement a capacity
limit, or does it persist regardless of network size?

Item 15 of the DQN roadmap ("use a sufficiently expressive fully connected
network") was, until now, an assertion: `_QNet`'s `5 -> 96 -> 96 -> 16`
architecture was never checked against anything bigger or deeper. The
warm-start experiment (`nash_dqn.py`) already showed that a *bigger training
budget* doesn't move action agreement -- fitting directly to `Q_exact` with
zero bootstrap noise only reaches 45%, same as 600 epochs of TD bootstrap
from scratch. This script asks the complementary question: does a *bigger
network* move it?

Uses `fit_q_to_exact` (plain supervised regression onto the exact stage
matrices, no bootstrap, no transition-table precompute) rather than the full
TD-bootstrap loop -- `nash_dqn.py` already established that the two training
methods land in the same place, and the supervised fit is far cheaper per
run, which is what makes sweeping many architectures affordable at all.

Two sweeps, both fixed at 600 epochs:

* **width**, depth fixed at 2 hidden layers (the original architecture):
  `hidden in {32, 64, 96, 128, 192, 256}`.
* **depth**, width fixed at 96: `hidden in {(96,), (96,96,96), (96,96,96,96)}`
  (depth 2 / width 96 is already the width sweep's `hidden=96` point).

Plus one combined "go big both ways" point, `hidden=(256, 256, 256)`.

2 seeds per configuration (an ablation grid trades seed count for breadth of
configurations; `nash_dqn.py`'s own headline numbers use 5).

    python scripts/nash_dqn_ablation.py           # writes experiments/nash_dqn_ablation.csv
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
from soccer_nash.nash_dqn import compare_to_exact, fit_q_to_exact
from soccer_nash.nash_q import NashQIteration

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "nash_dqn_ablation.csv"
FIELDS = [
    "sweep", "depth", "width", "n_params", "seed", "train_time_s",
    "max_value_error", "mean_value_error", "action_agreement",
    "classification_agreement", "duality_gap",
]


def _n_params(hidden: tuple[int, ...], out: int = 16) -> int:
    sizes = [5, *hidden, out]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def _configs() -> list[tuple[str, int, int, tuple[int, ...]]]:
    """(sweep_name, depth, width, hidden_tuple) -- width varies depth-2
    hidden layers; depth varies width-96 hidden layers; depth=2/width=96 is
    shared between both sweeps (labelled "width" since it's that sweep's
    midpoint) so it is listed once, not trained twice."""
    out = []
    for width in (32, 64, 96, 128, 192, 256):
        out.append(("width", 2, width, (width, width)))
    for depth in (1, 3, 4):
        out.append(("depth", depth, 96, (96,) * depth))
    out.append(("both", 3, 256, (256, 256, 256)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    game = A10SoccerGame()
    exact = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10).run()
    no_saddle = set(exact.no_saddle_states)
    solver = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10)
    solver.run()
    matrix_of = lambda s: solver._matrix(s, exact.values)  # noqa: E731

    configs = _configs()
    print(f"{len(configs)} architectures x {args.seeds} seeds x {args.epochs} "
          f"epochs, fit-to-exact (supervised, no bootstrap)\n")

    rows: list[dict] = []
    for sweep, depth, width, hidden in configs:
        n_params = _n_params(hidden)
        accs, exploits, verrs = [], [], []
        for seed in range(args.seeds):
            t = time.perf_counter()
            net = fit_q_to_exact(
                game, matrix_of, hidden=hidden, epochs=args.epochs, seed=seed
            )
            dt = time.perf_counter() - t
            m = compare_to_exact(
                game, net, exact.values, exact.row_policy, args.gamma, no_saddle
            )
            accs.append(m["action_agreement"])
            exploits.append(m["duality_gap"])
            verrs.append(m["max_value_error"])
            rows.append({
                "sweep": sweep, "depth": depth, "width": width,
                "n_params": n_params, "seed": seed,
                "train_time_s": round(dt, 1),
                "max_value_error": round(m["max_value_error"], 4),
                "mean_value_error": round(m["mean_value_error"], 4),
                "action_agreement": round(m["action_agreement"], 4),
                "classification_agreement": round(m["classification_agreement"], 4),
                "duality_gap": round(m["duality_gap"], 4),
            })
        print(f"  {sweep:5} depth={depth} width={width:>3} "
              f"({n_params:>6,} params)  agree={statistics.mean(accs):.3f}  "
              f"exploit={statistics.mean(exploits):.3f}  "
              f"max_verr={statistics.mean(verrs):.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    by_config: dict[tuple, list[dict]] = {}
    for r in rows:
        by_config.setdefault((r["sweep"], r["depth"], r["width"]), []).append(r)
    best = max(
        by_config.items(),
        key=lambda kv: statistics.mean(x["action_agreement"] for x in kv[1]),
    )
    baseline = by_config[("width", 2, 96)]
    baseline_acc = statistics.mean(x["action_agreement"] for x in baseline)
    best_acc = statistics.mean(x["action_agreement"] for x in best[1])
    print(f"\n=> baseline (depth=2, width=96): {baseline_acc:.3f} action agreement.")
    print(f"   best of the grid ({best[0][0]}, depth={best[0][1]}, "
          f"width={best[0][2]}): {best_acc:.3f}.")
    print("   policy net (nash_dqn.py, same game): 0.999 action agreement, "
          "for scale.")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
