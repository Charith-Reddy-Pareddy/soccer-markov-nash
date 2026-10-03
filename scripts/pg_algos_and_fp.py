"""Research-group-note items: (1) fictitious play vs best-response dynamics,
then FP on the soccer stage games; (2) REINFORCE vs A2C vs PPO self-play
against the exact solve.

    python scripts/pg_algos_and_fp.py fp                 # experiments/fictitious_play.csv
    python scripts/pg_algos_and_fp.py pg --seeds 3       # experiments/pg_algos_seeds.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.actor_critic import train_actor_critic_selfplay
from soccer_nash.exploit import best_response_to, onehot_policy, uniform_policy
from soccer_nash.fictitious_play import (
    best_response_dynamics,
    fictitious_play,
    markov_fictitious_play,
)
from soccer_nash.game import A10SoccerGame, SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.policy_gradient import evaluate_policy_gradient, train_reinforce_selfplay
from soccer_nash.winrate import mirror_gap, play_matches

EXP = pathlib.Path(__file__).resolve().parent.parent / "experiments"
RPS = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)


def run_fp(rounds_list=(100, 1000, 10000)) -> None:
    print("== rock-paper-scissors ==")
    path = best_response_dynamics(RPS, 30)
    print("best-response dynamics, first 12 (row, col) actions:", path[:12])
    print("  distinct joint states visited:", len(set(path)), "(a cycle: never settles)")
    for T in (100, 1000, 10000):
        p, q, lo, hi = fictitious_play(RPS, T)
        print(f"fictitious play T={T:>5}: p={np.round(p, 3)} q={np.round(q, 3)} "
              f"value in [{lo:+.4f}, {hi:+.4f}]")

    game = SoccerGame(move_order="random")
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    exact = solver.run_exact()
    states = list(game.states())
    Ms = np.array([solver._matrix(s, exact.values) for s in states])
    v_exact = np.array([exact.values[s] for s in states])
    # a state is mixed when maximin < minimax on its stage matrix
    mixed = Ms.min(2).max(1) < Ms.max(1).min(1) - 1e-9
    print(f"\n== fictitious play on the exact stage games ({len(states)} states, "
          f"{int(mixed.sum())} with no pure saddle, random move order) ==")
    rows = []
    for T in rounds_list:
        t = time.perf_counter()
        _, _, lo, hi = fictitious_play(Ms, T)
        dt = time.perf_counter() - t
        err = np.abs((lo + hi) / 2 - v_exact)
        row = {
            "rounds": T,
            "max_gap_all": float((hi - lo).max()),
            "mean_gap_all": float((hi - lo).mean()),
            "max_value_err_all": float(err.max()),
            "mean_gap_pure_states": float((hi - lo)[~mixed].mean()),
            "mean_gap_mixed_states": float((hi - lo)[mixed].mean()),
            "max_value_err_mixed": float(err[mixed].max()),
            "seconds": round(dt, 2),
        }
        rows.append(row)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    EXP.mkdir(exist_ok=True)
    with (EXP / "fictitious_play.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote experiments/fictitious_play.csv")


def run_markov_fp() -> None:
    game = SoccerGame(move_order="random")
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    exact = solver.run_exact()
    cps = (10, 50, 100, 200, 500, 1000, 2000)
    rows = []
    for persistent, label, sweeps, rounds in (
        (False, "restart-200", 150, 200), (True, "persistent", 2000, 0),
    ):
        t = time.perf_counter()
        _, hist = markov_fictitious_play(
            solver, exact.values, sweeps, persistent, rounds, checkpoints=cps)
        print(f"{label} ({time.perf_counter() - t:.0f}s):")
        for k, err, br in hist:
            print(f"  sweep {k:>5}: max|V-V*| = {err:.4f}, mean bracket = {br:.4f}")
            rows.append({"variant": label, "sweep": k, "max_value_err": err,
                         "mean_bracket": br})
    EXP.mkdir(exist_ok=True)
    with (EXP / "markov_fictitious_play.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote experiments/markov_fictitious_play.csv")


def run_pg(seeds: int, iterations: int, rollout_len: int, board: str, n_games: int) -> None:
    game = A10SoccerGame() if board == "a10" else SoccerGame(move_order="random")
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    exact = solver.run()
    matrix_of = lambda s: solver._matrix(s, exact.values)  # noqa: E731
    algos = {
        "reinforce": lambda sd: train_reinforce_selfplay(
            game, iterations=iterations, rollout_len=rollout_len, seed=sd,
            use_baseline=True, entropy_coef=0.01),
        "a2c": lambda sd: train_actor_critic_selfplay(
            game, "a2c", iterations=iterations, rollout_len=rollout_len, seed=sd),
        "ppo": lambda sd: train_actor_critic_selfplay(
            game, "ppo", iterations=iterations, rollout_len=rollout_len, seed=sd),
    }
    rows = []
    for name, fn in algos.items():
        for sd in range(seeds):
            t = time.perf_counter()
            res = fn(sd)
            dt = time.perf_counter() - t
            m = evaluate_policy_gradient(
                game, res, exact.row_policy, exact.col_policy, matrix_of)
            rp = {st: res.net0.policy(st) for st in game.states()}
            cp = {st: res.net1.policy(st) for st in game.states()}
            br_col = onehot_policy(
                best_response_to(game, rp, responder=1).policy, game.n_actions)
            opp = {"random": uniform_policy(game), "nash": exact.col_policy, "br": br_col}
            wl = {}
            for oname, op in opp.items():
                for k, v in play_matches(game, rp, op, n_games, seed=sd).items():
                    wl[f"row_{k}_vs_{oname}"] = v
            wl["mirror_gap_mean"], wl["mirror_gap_max"] = mirror_gap(game, rp, cp)
            rows.append({"algo": name, "seed": sd, "train_s": round(dt, 1), **{
                k: round(v, 4) for k, v in {**m, **wl}.items()}})
            print(name, sd, rows[-1], flush=True)
    EXP.mkdir(exist_ok=True)
    with (EXP / f"pg_algos_{board}_seeds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nboard: {board}")
    print("algo | exploitability | mean eq. regret | row agree | row vs random | row vs BR")
    for name in algos:
        r = [x for x in rows if x["algo"] == name]
        def f(k, r=r):
            return f"{statistics.mean(x[k] for x in r):.3f}"
        print(f"{name:9s} | {f('duality_gap')} | {f('mean_equilibrium_regret')} | "
              f"{f('row_action_agreement')} | {f('row_vs_random')} | {f('row_vs_best_response')}")
        for o in ("random", "nash", "br"):
            print(f"          vs {o:6s}: W/T/L = {f(f'row_win_vs_{o}')} / "
                  f"{f(f'row_tie_vs_{o}')} / {f(f'row_loss_vs_{o}')}")
        print(f"          mirror gap mean/max = {f('mirror_gap_mean')} / {f('mirror_gap_max')}")
    print(f"wrote experiments/pg_algos_{board}_seeds.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", choices=["fp", "pg", "markov-fp"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--board", choices=["random", "a10"], default="random")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--rollout-len", type=int, default=100)
    a = ap.parse_args()
    if a.which == "fp":
        run_fp()
    elif a.which == "markov-fp":
        run_markov_fp()
    else:
        run_pg(a.seeds, a.iterations, a.rollout_len, a.board, a.games)
