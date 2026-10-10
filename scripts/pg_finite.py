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


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", choices=["a10", "random"], default="a10")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0, help="first seed to run")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--algos", nargs="+", default=list(pf.ALGOS), choices=[*pf.ALGOS, "a2c_exact"])
    ap.add_argument("--scoring", choices=["win", "rate"], default="win",
                    help="win: the first goal ends the game; rate: play continues after a goal")
    ap.add_argument("--modes", nargs="+", default=["selfplay", "fictitious"], choices=pf.MODES)
    ap.add_argument("--entropy", type=float, default=0.01, help="entropy bonus")
    ap.add_argument("--shared", action="store_true", help="one network, one output head per player")
    ap.add_argument("--trim", type=int, default=0,
                    help="drop the last N steps of each episode from the loss")
    ap.add_argument("--resume", action="store_true",
                    help="keep the runs already in the output file and skip them")
    ap.add_argument("--prefix", default="pg_finite", help="start of the output file name")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    return ap


def main() -> None:
    a = parser().parse_args()

    order = "deterministic" if a.board == "a10" else "random"
    if a.scoring == "rate":
        game = SoccerGame(move_order=order, scoring="rate", max_steps=100)
    else:
        game = A10SoccerGame() if a.board == "a10" else SoccerGame(move_order="random")
    horizon = game.max_steps
    solver = NashQIteration(game, gamma=a.gamma, mode="hybrid", tol=1e-10)
    _, row_t, col_t = fh.solve_finite_horizon(solver, a.gamma, horizon)
    exact = (fh.tables_to_policy(row_t), fh.tables_to_policy(col_t))
    critic = fh.ExactCritic(game, solver, a.gamma, horizon) if "a2c_exact" in a.algos else None
    rows = [{"algo": "exact", "mode": "-", "seed": 0, "train_s": 0.0, "iterations": 0,
             **fh.evaluate(game, solver, exact, *exact, a.gamma, horizon, a.games, 0)}]
    print(rows[0], flush=True)
    name = a.board if a.scoring == "win" else f"rate_{a.board}"
    path = EXP / f"{a.prefix}_{name}{a.tag}.csv"

    done = set()
    if a.resume and path.exists():
        with path.open() as f:
            old_rows = [r for r in csv.DictReader(f) if r["algo"] != "exact"]
        rows.extend(old_rows)
        done = {(r["algo"], r["mode"], int(r["seed"])) for r in old_rows}

    def save() -> None:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    for algo in a.algos:
        for mode in a.modes:
            for seed in range(a.seed_start, a.seed_start + a.seeds):
                if (algo, mode, seed) in done:
                    continue
                t = time.perf_counter()
                tr = pf.train(game, algo, mode, a.gamma, horizon, a.iterations,
                              a.episodes, seed=seed, exact=critic,
                              shared=a.shared, trim=a.trim, entropy=a.entropy)
                dt = round(time.perf_counter() - t, 1)
                rows.append({"algo": algo, "mode": mode, "seed": seed, "train_s": dt,
                             "iterations": a.iterations,
                             **fh.evaluate(game, solver, exact, tr.pol0, tr.pol1,
                                        a.gamma, horizon, a.games, seed)})
                print(rows[-1], flush=True)
                save()
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
