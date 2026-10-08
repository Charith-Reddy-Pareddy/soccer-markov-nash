"""How the policy-gradient learners behave at the states where the exact equilibrium has to
mix (random move-order board): win rates from games started at those states, and how close
their action probabilities are to the exact mix.

    python scripts/pg_mixed_states.py --algo a2c --mode selfplay     # 3 seeds -> one JSON
    python scripts/pg_mixed_states.py --merge                        # combine, add the exact solver
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash import finite_horizon as fh
from soccer_nash import pg_finite as pf
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
GAMES_PER_START = 20
EXAMPLES = 3


def setup(gamma: float):
    game = SoccerGame(move_order="random")
    horizon = game.max_steps
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-10)
    values, row_t, col_t = fh.solve_finite_horizon(solver, gamma, horizon)
    mixed = fh.mixed_states(solver, values, gamma)
    exact = (fh.tables_to_policy(row_t), fh.tables_to_policy(col_t))
    return game, horizon, solver, values, row_t, col_t, mixed, exact


def start_results(game, solver, exact, row, col, mixed, horizon, gamma) -> dict:
    n = GAMES_PER_START * len(mixed)
    _, br_table = fh.best_response(solver, col, 1, gamma, horizon)
    opp = {"random": fh.uniform, "nash": exact[1], "br": fh.br_policy(br_table)}
    return {k: [round(v, 6) for v in fh.play(game, row, o, n, horizon, 0, starts=mixed).values()]
            for k, o in opp.items()}


def examples(mixed, exact_row0, exact_col0) -> list:
    """Three mixed states to show: the first with a three-way mix, then the first two-way mixes."""
    three = [s for s in mixed if (exact_row0[s] > 1e-6).sum() >= 3]
    two = [s for s in mixed if (exact_row0[s] > 1e-6).sum() == 2]
    return (three[:1] + two[:EXAMPLES - len(three[:1])])[:EXAMPLES]


def run(algo: str, mode: str, seeds: int, gamma: float, iterations: int) -> None:
    game, horizon, solver, values, row_t, col_t, mixed, exact = setup(gamma)
    states = solver._states
    index = {s: i for i, s in enumerate(states)}
    ex_row = {s: row_t[0][s] for s in states}
    ex_col = {s: col_t[0][s] for s in states}
    shown = examples(mixed, ex_row, ex_col)
    records = []
    for seed in range(seeds):
        tr = pf.train(game, algo, mode, gamma, horizon, iterations, 64, seed=seed)
        r0, c0 = tr.pol0(0, states), tr.pol1(0, states)
        row0 = {s: r0[index[s]] for s in states}
        col0 = {s: c0[index[s]] for s in states}
        rec = {"algo": algo, "mode": mode, "seed": seed}
        rec.update(fh.mixed_state_metrics(solver, values, gamma, mixed, row0, col0, ex_row, ex_col))
        rec["wins_from_mixed_starts"] = start_results(
            game, solver, exact, tr.pol0, tr.pol1, mixed, horizon, gamma)
        rec["examples"] = [{"state": list(s), "row": row0[s].round(4).tolist(),
                            "col": col0[s].round(4).tolist()} for s in shown]
        records.append(rec)
        brief = {k: round(v, 3) for k, v in rec.items() if isinstance(v, float)}
        print(algo, mode, seed, brief, flush=True)
        (EXP / f"pg_mixed_{algo}_{mode}.json").write_text(json.dumps(records, indent=1) + "\n")


def merge(gamma: float) -> None:
    game, horizon, solver, values, row_t, col_t, mixed, exact = setup(gamma)
    ex_row = {s: row_t[0][s] for s in solver._states}
    ex_col = {s: col_t[0][s] for s in solver._states}
    n = GAMES_PER_START * len(mixed)
    opponents = {"random": fh.uniform, "nash": exact[1], "br": exact[1]}
    wins = {}
    for k, o in opponents.items():
        res = fh.play(game, exact[0], o, n, horizon, 0, starts=mixed)
        wins[k] = [round(v, 6) for v in res.values()]
    shown = examples(mixed, ex_row, ex_col)
    sizes = {str(k): sum(1 for s in mixed if (ex_row[s] > 1e-6).sum() == k) for k in (1, 2, 3, 4)}
    runs = []
    for path in sorted(glob.glob(str(EXP / "pg_mixed_*_*.json"))):
        runs += json.loads(pathlib.Path(path).read_text())
    out = {"board": "random", "mixed_states": len(mixed), "games_per_start": GAMES_PER_START,
           "exact": {"wins_from_mixed_starts": wins, "support_sizes": sizes},
           "examples": [{"state": list(s), "row": ex_row[s].round(4).tolist(),
                         "col": ex_col[s].round(4).tolist()} for s in shown],
           "runs": runs}
    (EXP / "pg_mixed_states.json").write_text(json.dumps(out, indent=1) + "\n")
    print("merged", len(runs), "runs;", len(mixed), "mixed states")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", choices=list(pf.ALGOS))
    ap.add_argument("--mode", choices=["selfplay", "fictitious"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge(a.gamma) if a.merge else run(a.algo, a.mode, a.seeds, a.gamma, a.iterations)
