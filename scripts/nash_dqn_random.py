"""Neural Nash-Q on the game where mixed equilibria actually matter.

`scripts/nash_dqn.py` runs all of this on `A10SoccerGame` -- the
**deterministic** game, where every stage game already has a pure saddle.
That made the from-zero/fit-to-exact/warm-start comparison a genuine
function-approximation study, but not yet an answer to the harder question:

    Can neural Nash-Q reproduce the exact solver on a game where mixed
    equilibria actually matter?

This script runs the same three Q-net starting points, plus the policy net,
on the plain 4x4 board under Littman's **random move order** (the same board
as `tournament4.py`/`tournament_deepdive.py`: 5x4, two-cell goals, no
`stand`) -- 56 of 760 states have no pure saddle. That needed two real fixes
in `soccer_nash/nash_dqn.py`, not just a different game object:

- `train_nash_dqn`'s bootstrap target used to be the vectorised *maximin*
  of the target network's predicted matrix -- exact only when every stage
  game has a pure saddle. It is now `_minimax_batch`: the same pure-saddle
  fast path, vectorised, falling back to the LP (`game_value`) only for the
  predicted-mixed matrices -- the general hybrid solver's own logic, applied
  to the *network's* predictions each epoch.
- the transition precompute used to call `game.step`, which *samples* one
  outcome -- exact only when a joint action has a single outcome, i.e. only
  on the deterministic game. It now uses `game.transitions`, the full
  distribution, and averages over it exactly.

Metrics now go beyond action agreement, per the concern that argmax accuracy
can look better than it is in a game with ties and non-unique equilibria:
mean/max Frobenius (matrix) error, mean/max equilibrium regret against the
*exact* stage matrix (not the net's own), and support agreement.

    python scripts/nash_dqn_random.py --seeds 5

Writes `experiments/nash_dqn_random_seeds.csv`.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_dqn import (
    compare_policy_to_exact,
    compare_to_exact,
    fit_q_to_exact,
    train_nash_dqn,
    train_policy_baseline,
)
from soccer_nash.nash_q import NashQIteration

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "nash_dqn_random_seeds.csv"
BOARD = {"width": 5, "height": 4, "goal_rows": (1, 2), "move_order": "random",
         "n_actions": 4, "scoring": "rate"}
FIELDS = [
    "seed",
    "from_zero_time_s", "from_zero_action_agreement", "from_zero_duality_gap",
    "from_zero_mean_frobenius_error", "from_zero_mean_equilibrium_regret",
    "from_zero_row_support_agreement", "from_zero_col_support_agreement",
    "fit_time_s", "fit_action_agreement", "fit_duality_gap",
    "fit_mean_frobenius_error", "fit_mean_equilibrium_regret",
    "fit_row_support_agreement", "fit_col_support_agreement",
    "warm_time_s", "warm_action_agreement", "warm_duality_gap",
    "warm_mean_frobenius_error", "warm_mean_equilibrium_regret",
    "warm_row_support_agreement", "warm_col_support_agreement",
    "policy_action_agreement", "policy_max_regret", "policy_duality_gap",
]


def _fmt(xs: list[float], digits: int = 3) -> str:
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    game = SoccerGame(**BOARD)
    t = time.perf_counter()
    exact = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10).run()
    t_exact = time.perf_counter() - t
    no_saddle = set(exact.no_saddle_states)
    solver = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10)
    solver.run()
    matrix_of = lambda s: solver._matrix(s, exact.values)  # noqa: E731
    n_states = len(list(game.states()))

    print(f"plain 4x4 board (5x4, 2-cell goals, RANDOM move order), "
          f"gamma {args.gamma}")
    print(f"{len(no_saddle)} of {n_states} states "
          f"({100 * len(no_saddle) / n_states:.1f}%) have no pure saddle -- "
          f"this is the game the deterministic-game DQN experiment "
          f"(nash_dqn.py) never actually exercised.\n")

    rows: list[dict] = []
    for seed in range(args.seeds):
        t = time.perf_counter()
        dqn = train_nash_dqn(
            game, gamma=args.gamma, hidden=args.hidden, epochs=args.epochs, seed=seed
        )
        t_zero = time.perf_counter() - t
        m = compare_to_exact(
            game, dqn.net, exact.values, exact.row_policy, args.gamma, no_saddle,
            exact_matrix_of=matrix_of, exact_col_policy=exact.col_policy,
        )

        t = time.perf_counter()
        qfit = fit_q_to_exact(
            game, matrix_of, hidden=args.hidden, epochs=args.epochs, seed=seed
        )
        t_fit = time.perf_counter() - t
        fm = compare_to_exact(
            game, qfit, exact.values, exact.row_policy, args.gamma, no_saddle,
            exact_matrix_of=matrix_of, exact_col_policy=exact.col_policy,
        )

        t = time.perf_counter()
        warm = train_nash_dqn(
            game, gamma=args.gamma, hidden=args.hidden, epochs=args.epochs,
            seed=seed, init_net=qfit,
        )
        t_warm = time.perf_counter() - t
        wm = compare_to_exact(
            game, warm.net, exact.values, exact.row_policy, args.gamma, no_saddle,
            exact_matrix_of=matrix_of, exact_col_policy=exact.col_policy,
        )

        pnet = train_policy_baseline(
            game, exact.row_policy, exact.col_policy, hidden=args.hidden,
            epochs=args.epochs, seed=seed,
        )
        pm = compare_policy_to_exact(
            game, pnet, matrix_of, exact.row_policy, no_saddle, args.gamma
        )

        rows.append({
            "seed": seed,
            "from_zero_time_s": round(t_zero, 1),
            "from_zero_action_agreement": round(m["action_agreement"], 4),
            "from_zero_duality_gap": round(m["duality_gap"], 4),
            "from_zero_mean_frobenius_error": round(m["mean_frobenius_error"], 4),
            "from_zero_max_entrywise_error": round(m["max_entrywise_error"], 4),
            "from_zero_mean_equilibrium_regret": round(m["mean_equilibrium_regret"], 4),
            "from_zero_max_equilibrium_regret": round(m["max_equilibrium_regret"], 4),
            "from_zero_row_support_agreement": round(m["row_support_agreement"], 4),
            "from_zero_col_support_agreement": round(m["col_support_agreement"], 4),
            "fit_time_s": round(t_fit, 1),
            "fit_action_agreement": round(fm["action_agreement"], 4),
            "fit_duality_gap": round(fm["duality_gap"], 4),
            "fit_mean_frobenius_error": round(fm["mean_frobenius_error"], 4),
            "fit_max_entrywise_error": round(fm["max_entrywise_error"], 4),
            "fit_mean_equilibrium_regret": round(fm["mean_equilibrium_regret"], 4),
            "fit_max_equilibrium_regret": round(fm["max_equilibrium_regret"], 4),
            "fit_row_support_agreement": round(fm["row_support_agreement"], 4),
            "fit_col_support_agreement": round(fm["col_support_agreement"], 4),
            "warm_time_s": round(t_warm, 1),
            "warm_action_agreement": round(wm["action_agreement"], 4),
            "warm_duality_gap": round(wm["duality_gap"], 4),
            "warm_mean_frobenius_error": round(wm["mean_frobenius_error"], 4),
            "warm_max_entrywise_error": round(wm["max_entrywise_error"], 4),
            "warm_mean_equilibrium_regret": round(wm["mean_equilibrium_regret"], 4),
            "warm_max_equilibrium_regret": round(wm["max_equilibrium_regret"], 4),
            "warm_row_support_agreement": round(wm["row_support_agreement"], 4),
            "warm_col_support_agreement": round(wm["col_support_agreement"], 4),
            "policy_action_agreement": round(pm["action_agreement"], 4),
            "policy_max_regret": round(pm["max_equilibrium_regret"], 4),
            "policy_duality_gap": round(pm["duality_gap"], 4),
        })
        print(f"  seed {seed}: from-zero agree {rows[-1]['from_zero_action_agreement']:.3f} "
              f"exploit {rows[-1]['from_zero_duality_gap']:.3f}  |  "
              f"fit agree {rows[-1]['fit_action_agreement']:.3f} "
              f"exploit {rows[-1]['fit_duality_gap']:.3f}  |  "
              f"warm agree {rows[-1]['warm_action_agreement']:.3f} "
              f"exploit {rows[-1]['warm_duality_gap']:.3f}  |  "
              f"pi-net agree {rows[-1]['policy_action_agreement']:.3f} "
              f"exploit {rows[-1]['policy_duality_gap']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nexact hybrid Nash-Q : {exact.iterations} sweeps, {t_exact:.2f} s, "
          f"value error 0, exploitability 0\n")
    for label, prefix in (("from zero", "from_zero"), ("fit to exact", "fit"),
                          ("warm start", "warm")):
        print(f"Q net, {label}:")
        print(f"  action agreement        : "
              f"{_fmt([r[f'{prefix}_action_agreement'] for r in rows])}")
        print(f"  exploitability           : "
              f"{_fmt([r[f'{prefix}_duality_gap'] for r in rows])}")
        print(f"  mean Frobenius error     : "
              f"{_fmt([r[f'{prefix}_mean_frobenius_error'] for r in rows])}")
        print(f"  mean equilibrium regret  : "
              f"{_fmt([r[f'{prefix}_mean_equilibrium_regret'] for r in rows])}")
        print(f"  row/col support agreement: "
              f"{_fmt([r[f'{prefix}_row_support_agreement'] for r in rows])} / "
              f"{_fmt([r[f'{prefix}_col_support_agreement'] for r in rows])}")
        print()
    print("policy net:")
    print(f"  action agreement : {_fmt([r['policy_action_agreement'] for r in rows])}")
    print(f"  exploitability    : {_fmt([r['policy_duality_gap'] for r in rows])}")
    print(f"\nwrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
