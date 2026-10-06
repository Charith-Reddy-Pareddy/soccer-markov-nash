"""REINFORCE / A2C / PPO on the discounted 100-step soccer game, by self-play
and by fictitious play, scored against the exact finite-horizon solution.

    python scripts/pg_finite.py --seeds 3            # A10 deterministic board
    python scripts/pg_finite.py --board random       # random move-order board
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash import finite_horizon as fh
from soccer_nash import pg_finite as pf
from soccer_nash.game import A10SoccerGame, SoccerGame
from soccer_nash.nash_q import NashQIteration

EXP = pathlib.Path(__file__).resolve().parent.parent / "experiments"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", choices=["a10", "random"], default="a10")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--algos", nargs="+", default=list(pf.ALGOS))
    ap.add_argument("--modes", nargs="+", default=["selfplay", "fictitious"])
    ap.add_argument("--tag", default="", help="suffix for the output file")
    a = ap.parse_args()

    game = A10SoccerGame() if a.board == "a10" else SoccerGame(move_order="random")
    horizon = game.max_steps
    solver = NashQIteration(game, gamma=a.gamma, mode="hybrid", tol=1e-10)
    _, row_t, col_t = fh.solve_finite_horizon(solver, a.gamma, horizon)
    exact = (fh.tables_to_policy(row_t), fh.tables_to_policy(col_t))
    rows = [{"algo": "exact", "mode": "-", "seed": 0, "train_s": 0.0,
             **fh.evaluate(game, solver, exact, *exact, a.gamma, horizon, a.games, 0)}]
    print(rows[0], flush=True)
    for algo in a.algos:
        for mode in a.modes:
            for seed in range(a.seeds):
                t = time.perf_counter()
                tr = pf.train(game, algo, mode, a.gamma, horizon, a.iterations,
                              a.episodes, seed=seed)
                dt = round(time.perf_counter() - t, 1)
                rows.append({"algo": algo, "mode": mode, "seed": seed, "train_s": dt,
                             **fh.evaluate(game, solver, exact, tr.pol0, tr.pol1,
                                        a.gamma, horizon, a.games, seed)})
                print(rows[-1], flush=True)
    path = EXP / f"pg_finite_{a.board}{a.tag}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nalgo | mode | exploitability | row wins vs random / nash / br | mirror gap")
    for algo in a.algos:
        for mode in a.modes:
            r = [x for x in rows if x["algo"] == algo and x["mode"] == mode]
            m = lambda k, r=r: statistics.mean(x[k] for x in r)  # noqa: E731
            print(f"{algo:9s} {mode:10s} {m('exploitability'):.3f}  "
                  f"{m('row_win_vs_random'):.3f} / {m('row_win_vs_nash'):.3f} / "
                  f"{m('row_win_vs_br'):.3f}  mirror {m('mirror_gap_mean'):.3f}")
    print(f"wrote experiments/{path.name}")


if __name__ == "__main__":
    main()
