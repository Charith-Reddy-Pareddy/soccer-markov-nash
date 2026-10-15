from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soccer_nash import finite_horizon as fh
from soccer_nash import pg_finite as pf
from soccer_nash.game import A10SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.finite_horizon import mirror_gap


OUT = ROOT / "experiments"
ALGOS = ("a2c", "a2c_mc", "a2c_pretrained")


def fast_br(game, solver, opponent, responder, gamma, horizon):
    """Exact finite-horizon BR using vectorized deterministic transitions."""
    states = solver._states
    ids = {s: i for i, s in enumerate(states)}
    n = len(states)
    nxt = np.zeros((n, 4, 4), dtype=np.int32)
    reward = np.zeros((n, 4, 4), dtype=np.float64)
    terminal = np.zeros((n, 4, 4), dtype=bool)
    for si, state in enumerate(states):
        for a0 in range(4):
            for a1 in range(4):
                outcomes = solver._out[state][a0, a1]
                if len(outcomes) != 1 or outcomes[0][0] != 1.0:
                    raise ValueError("fast_br currently requires deterministic transitions")
                _, ns, r0 = outcomes[0]
                is_terminal = game.is_terminal(ns)
                nxt[si, a0, a1] = 0 if is_terminal else ids[ns]
                reward[si, a0, a1] = r0
                terminal[si, a0, a1] = is_terminal

    sign = 1.0 if responder == 0 else -1.0
    values = np.zeros(n, dtype=np.float64)
    table = [{} for _ in range(horizon)]
    for t in reversed(range(horizon)):
        q = sign * reward + gamma * np.where(terminal, 0.0, values[nxt])
        opp = opponent(t, states)
        if responder == 0:
            action_vals = np.einsum("sij,sj->si", q, opp)
        else:
            action_vals = np.einsum("si,sij->sj", opp, q)
        acts = np.argmax(action_vals, axis=1)
        values = action_vals[np.arange(n), acts]
        table[t] = {s: int(a) for s, a in zip(states, acts)}
    return float(values[ids[game.initial_state()]]), table


def evaluate_fast(game, solver, exact, row, col, gamma, horizon, games, seed):
    row_value, row_br = fast_br(game, solver, col, 0, gamma, horizon)
    col_value, col_br = fast_br(game, solver, row, 1, gamma, horizon)
    out = {"exploitability": round(row_value + col_value, 4)}
    br_row, br_col = fh.br_policy(col_br), fh.br_policy(row_br)
    for label, opp in (("random", fh.uniform), ("nash", exact[1]), ("br", br_row)):
        match = fh.play(game, row, opp, games, horizon, seed)
        out.update({f"row_win_vs_{label}": match["win"],
                    f"row_tie_vs_{label}": match["tie"],
                    f"row_loss_vs_{label}": match["loss"]})
    for label, opp in (("random", fh.uniform), ("nash", exact[0]), ("br", br_col)):
        match = fh.play(game, opp, col, games, horizon, seed)
        out.update({f"col_win_vs_{label}": match["loss"],
                    f"col_tie_vs_{label}": match["tie"],
                    f"col_loss_vs_{label}": match["win"]})
    out["mirror_gap_mean"], out["mirror_gap_max"] = mirror_gap(game, row, col)
    return {k: round(v, 3) for k, v in out.items()}


def exact_critic(game, solver, gamma, horizon, values):
    """Build the professor-style frozen critic from the already solved values."""
    import torch

    critic = object.__new__(fh.ExactCritic)
    critic.width, critic.height, critic.horizon = game.width, game.height, horizon
    critic.states = tuple(solver._states)
    size = game.width * game.height * game.width * game.height * 2
    q = np.zeros((horizon, size, 4, 4), dtype=np.float32)
    v = np.zeros((horizon, size), dtype=np.float32)
    for state in solver._states:
        code = critic.code(*state)
        for t in range(horizon):
            q[t, code] = solver._matrix(state, values[t + 1], gamma=gamma)
            v[t, code] = values[t][state]
    critic.Q, critic.V = torch.from_numpy(q), torch.from_numpy(v)
    return critic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired A10 A2C critic comparisons.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 11], help="seed IDs to run (default: 10 11)")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="CSV output path; use a new path to rerun recorded seeds")
    args = parser.parse_args()
    game = A10SoccerGame()
    gamma, horizon, games, updates = 0.9, 100, 1000, 5
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-10)
    print("solving exact 100-step reference", flush=True)
    values, row_t, col_t = fh.solve_finite_horizon(solver, gamma, horizon)
    exact = (fh.tables_to_policy(row_t), fh.tables_to_policy(col_t))
    teacher = None
    OUT.mkdir(parents=True, exist_ok=True)
    path = args.output or OUT / "pg_paired_new_seeds_a10.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open() as f:
            rows = list(csv.DictReader(f))
    done = {(r["algo"], int(r["seed"])) for r in rows}
    for algo in ALGOS:
        for seed in args.seeds:
            if (algo, seed) in done:
                continue
            start = time.perf_counter()
            print(f"training {algo} seed {seed}", flush=True)
            if algo == "a2c_pretrained" and teacher is None:
                print("precomputing exact critic values", flush=True)
                teacher = exact_critic(game, solver, gamma, horizon, values)
            tr = pf.train(
                game, algo, mode="fictitious", gamma=gamma, horizon=horizon,
                iterations=2000, episodes=64, seed=seed,
                exact=teacher if algo == "a2c_pretrained" else None,
                entropy=0.01, snap_every=5, updates_per_batch=updates,
                critic_pretrain_epochs=10, critic_pretrain_batch_size=4096,
            )
            metrics = evaluate_fast(
                game, solver, exact, tr.pol0, tr.pol1, gamma, horizon, games, seed,
            )
            row = {
                "algo": algo,
                "seed": seed,
                "train_s": round(time.perf_counter() - start, 1),
                "iterations": 2000,
                "episodes": 64,
                "updates_per_batch": updates,
                "snap_every": 5,
                "entropy": 0.01,
                "critic_pretrain_epochs": 10 if algo == "a2c_pretrained" else 0,
                "games_per_seat": games,
                **metrics,
            }
            rows.append(row)
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader()
                w.writerows(rows)
            print(row, flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
