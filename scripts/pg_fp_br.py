"""Fictitious play with best-response phases, by policy gradient, on the discounted
100-step A10 soccer game; prints and saves exploitability after every few rounds.

    python scripts/pg_fp_br.py --algo a2c --seed 0
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash import finite_horizon as fh
from soccer_nash import pg_finite as pf
from soccer_nash.game import A10SoccerGame
from soccer_nash.nash_q import NashQIteration

EXP = pathlib.Path(__file__).resolve().parent.parent / "experiments"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", choices=list(pf.ALGOS), default="a2c")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--br-iters", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    game = A10SoccerGame()
    horizon = game.max_steps
    solver = NashQIteration(game, gamma=a.gamma, mode="hybrid", tol=1e-10)
    _, row_t, col_t = fh.solve_finite_horizon(solver, a.gamma, horizon)
    exact = (fh.tables_to_policy(row_t), fh.tables_to_policy(col_t))
    path = EXP / f"pg_fp_br_{a.algo}_s{a.seed}{a.tag}.csv"
    rows: list[dict] = []
    t0 = time.perf_counter()

    def on_round(r: int, tr: pf.Trained) -> None:
        if r % a.eval_every and r != a.rounds:
            return
        row = {"algo": a.algo, "seed": a.seed, "round": r,
               "br_iters": a.br_iters, "elapsed_s": round(time.perf_counter() - t0),
               **fh.evaluate(game, solver, exact, tr.pol0, tr.pol1, a.gamma, horizon,
                             a.games, a.seed)}
        rows.append(row)
        print(row, flush=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    pf.train_fp_br(game, a.algo, a.rounds, a.br_iters, a.gamma, horizon, a.episodes,
                   a.lr, seed=a.seed, on_round=on_round)
    print(f"wrote experiments/{path.name}")


if __name__ == "__main__":
    main()
