"""Does pre-training help self-play REINFORCE? -- an open question from the
Sept 17 meeting ("would pre-training help? ... minor issue", left for later).

Three starting points, the same shape as ``scripts/nash_dqn.py``'s
from-zero / fit-to-exact / warm-start comparison, adapted to REINFORCE's
on-policy loop instead of TD bootstrap:

* **from scratch** -- random init, straight into self-play REINFORCE.
* **fit to exact** -- supervised cross-entropy regression onto the exact
  equilibrium policies, no rollouts, no self-play at all. Isolates how well
  a network this size can represent the exact policy before any on-policy
  noise gets involved.
* **warm start** -- the fit-to-exact weights, then the *same* self-play
  REINFORCE loop as "from scratch", for the same number of iterations. Does
  starting already close to the answer survive further on-policy training,
  or does the policy-gradient noise erase the head start the way
  ``nash_dqn.py`` found TD bootstrap does?

    python scripts/policy_gradient_warmstart.py --seeds 3
    # writes experiments/policy_gradient_warmstart.csv
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
    ReinforceResult,
    evaluate_policy_gradient,
    pretrain_policy_nets,
    train_reinforce_selfplay,
)

OUT = (pathlib.Path(__file__).resolve().parent.parent
       / "experiments" / "policy_gradient_warmstart.csv")
FIELDS = [
    "seed",
    "scratch_row_agree", "scratch_duality_gap", "scratch_row_vs_br",
    "fit_row_agree", "fit_duality_gap", "fit_row_vs_br",
    "warm_row_agree", "warm_duality_gap", "warm_row_vs_br",
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
    parser.add_argument("--pretrain-epochs", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    game = A10SoccerGame()

    exact = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10).run()
    solver = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10)
    solver.run()
    matrix_of = lambda s: solver._matrix(s, exact.values)  # noqa: E731

    def _score(result):
        m = evaluate_policy_gradient(
            game, result, exact.row_policy, exact.col_policy, matrix_of, gamma=args.gamma,
        )
        return m["row_action_agreement"], m["duality_gap"], m["row_vs_best_response"]

    rows: list[dict] = []
    for seed in range(args.seeds):
        t = time.perf_counter()
        scratch = train_reinforce_selfplay(
            game, gamma=args.gamma, hidden=args.hidden, iterations=args.iterations,
            rollout_len=args.rollout_len, lr=args.lr, seed=seed,
        )
        t_scratch = time.perf_counter() - t
        s_agree, s_gap, s_br = _score(scratch)

        t = time.perf_counter()
        fit_net0, fit_net1 = pretrain_policy_nets(
            game, exact.row_policy, exact.col_policy, hidden=args.hidden,
            epochs=args.pretrain_epochs, seed=seed,
        )
        t_fit = time.perf_counter() - t
        fit_result = ReinforceResult(net0=fit_net0, net1=fit_net1, mean_reward_trace=[])
        f_agree, f_gap, f_br = _score(fit_result)

        t = time.perf_counter()
        warm = train_reinforce_selfplay(
            game, gamma=args.gamma, hidden=args.hidden, iterations=args.iterations,
            rollout_len=args.rollout_len, lr=args.lr, seed=seed,
            init_net0=fit_net0, init_net1=fit_net1,
        )
        t_warm = time.perf_counter() - t
        w_agree, w_gap, w_br = _score(warm)

        rows.append({
            "seed": seed,
            "scratch_row_agree": round(s_agree, 4),
            "scratch_duality_gap": round(s_gap, 4),
            "scratch_row_vs_br": round(s_br, 4),
            "fit_row_agree": round(f_agree, 4),
            "fit_duality_gap": round(f_gap, 4),
            "fit_row_vs_br": round(f_br, 4),
            "warm_row_agree": round(w_agree, 4),
            "warm_duality_gap": round(w_gap, 4),
            "warm_row_vs_br": round(w_br, 4),
        })
        print(f"  seed {seed} ({t_scratch:.0f}s/{t_fit:.0f}s/{t_warm:.0f}s): "
              f"scratch agree {s_agree:.3f} exploit {s_gap:.3f}  |  "
              f"fit-to-exact agree {f_agree:.3f} exploit {f_gap:.3f}  |  "
              f"warm agree {w_agree:.3f} exploit {w_gap:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nfrom scratch  -- row agreement {_fmt([r['scratch_row_agree'] for r in rows])}"
          f"  exploit {_fmt([r['scratch_duality_gap'] for r in rows])}")
    print(f"fit to exact  -- row agreement {_fmt([r['fit_row_agree'] for r in rows])}"
          f"  exploit {_fmt([r['fit_duality_gap'] for r in rows])}")
    print(f"warm start    -- row agreement {_fmt([r['warm_row_agree'] for r in rows])}"
          f"  exploit {_fmt([r['warm_duality_gap'] for r in rows])}")

    fit_agree = statistics.mean(r["fit_row_agree"] for r in rows)
    warm_agree = statistics.mean(r["warm_row_agree"] for r in rows)
    scratch_agree = statistics.mean(r["scratch_row_agree"] for r in rows)
    verdict = (
        "the self-play loop erases almost all of the warm start's head start"
        if warm_agree - scratch_agree < 0.5 * (fit_agree - scratch_agree)
        else "the warm start holds: most of its head start survives continued self-play"
    )
    print(f"\n=> fit-to-exact starts at {fit_agree:.3f} row action agreement; "
          f"{args.iterations} more iterations of the same self-play REINFORCE that "
          f"produces {scratch_agree:.3f} from a random init leaves the warm-started "
          f"net at {warm_agree:.3f} -- {verdict}.")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
