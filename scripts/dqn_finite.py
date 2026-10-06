"""Nash-DQN on the discounted 100-step soccer game, scored like the
policy-gradient learners (exploitability, win / tie / loss counts, mirror gap).

    python scripts/dqn_finite.py --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash import finite_horizon as fh
from soccer_nash.dqn_finite import QPolicies, train_dqn
from soccer_nash.game import A10SoccerGame
from soccer_nash.nash_q import NashQIteration

EXP = pathlib.Path(__file__).resolve().parent.parent / "experiments"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=0.9)
    a = ap.parse_args()

    game = A10SoccerGame()
    horizon = game.max_steps
    solver = NashQIteration(game, gamma=a.gamma, mode="hybrid", tol=1e-10)
    _, row_t, col_t = fh.solve_finite_horizon(solver, a.gamma, horizon)
    exact = (fh.tables_to_policy(row_t), fh.tables_to_policy(col_t))
    rows = []
    for seed in range(a.seeds):
        t = time.perf_counter()
        net = train_dqn(solver, a.gamma, horizon, steps=a.steps, seed=seed)
        dt = round(time.perf_counter() - t, 1)
        q = QPolicies(solver, net, horizon)
        rows.append({"algo": "dqn", "mode": "-", "seed": seed, "train_s": dt,
                     **fh.evaluate(game, solver, exact, q.row, q.col,
                                   a.gamma, horizon, a.games, seed)})
        print(rows[-1], flush=True)
    path = EXP / "dqn_finite_a10.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote experiments/{path.name}")


if __name__ == "__main__":
    main()
