"""Does training converge in soccer? Every ``--every`` iterations, score the policy the learner
would report (the current networks for standard training, the average of snapshots for fictitious
play) and record its win rate against the exact best response and the exact Nash policy, plus the
probability of each move at the kickoff for the current network and for the reported policy.

    python scripts/pg_curves.py --board random --algos a2c ppo --modes selfplay fictitious
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash import finite_horizon as fh
from soccer_nash import pg_finite as pf
from soccer_nash.game import A10SoccerGame, SoccerGame
from soccer_nash.nash_q import NashQIteration

EXP = pathlib.Path(__file__).resolve().parent.parent / "experiments"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", choices=["a10", "random"], default="random")
    ap.add_argument("--algos", nargs="+", default=["a2c", "ppo"], choices=pf.ALGOS)
    ap.add_argument("--modes", nargs="+", default=["selfplay", "fictitious"], choices=pf.MODES)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true")
    return ap


def main() -> None:
    a = parser().parse_args()
    game = A10SoccerGame() if a.board == "a10" else SoccerGame(move_order="random")
    horizon = game.max_steps
    solver = NashQIteration(game, gamma=a.gamma, mode="hybrid", tol=1e-10)
    _, row_t, col_t = fh.solve_finite_horizon(solver, a.gamma, horizon)
    exact = (fh.tables_to_policy(row_t), fh.tables_to_policy(col_t))
    kick = game.initial_state()
    path = EXP / f"pg_curves_{a.board}{a.tag}.csv"
    rows: list[dict] = []
    done: set = set()
    if a.resume and path.exists():
        with path.open() as f:
            rows = list(csv.DictReader(f))
        done = {(r["algo"], r["mode"], int(r["seed"])) for r in rows}

    def save() -> None:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    for algo in a.algos:
        for mode in a.modes:
            for seed in range(a.seeds):
                if (algo, mode, seed) in done:
                    continue
                batch: list[dict] = []

                def on_checkpoint(it, reported, current, batch=batch, seed=seed, algo=algo,
                                  mode=mode):
                    score = fh.evaluate(game, solver, exact, reported.pol0, reported.pol1,
                                        a.gamma, horizon, a.games, seed)
                    agg = reported.pol0(0, [kick])[0]
                    cur = current.pol0(0, [kick])[0]
                    row = {"algo": algo, "mode": mode, "seed": seed, "iteration": it,
                           "win_br": score["row_win_vs_br"], "win_nash": score["row_win_vs_nash"],
                           "tie_nash": score["row_tie_vs_nash"],
                           "win_random": score["row_win_vs_random"]}
                    row.update({f"agg_{k}": round(float(v), 4) for k, v in enumerate(agg)})
                    row.update({f"cur_{k}": round(float(v), 4) for k, v in enumerate(cur)})
                    batch.append(row)

                t = time.perf_counter()
                pf.train(game, algo, mode, a.gamma, horizon, a.iterations, 64, seed=seed,
                         checkpoint_every=a.every, on_checkpoint=on_checkpoint)
                rows.extend(batch)
                save()
                print(algo, mode, seed, f"{time.perf_counter() - t:.0f}s",
                      np.round([r["win_br"] for r in batch], 2).tolist(), flush=True)
    print(f"wrote experiments/{path.name}")


if __name__ == "__main__":
    main()
