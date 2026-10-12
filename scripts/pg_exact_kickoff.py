"""The exact solver's move probabilities at the first step from the kickoff, for the training-curve
plots: where the learners' kickoff policy should end up.

    python scripts/pg_exact_kickoff.py     # experiments/pg_exact_kickoff.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash import finite_horizon as fh
from soccer_nash.game import A10SoccerGame, SoccerGame
from soccer_nash.nash_q import NashQIteration

EXP = pathlib.Path(__file__).resolve().parent.parent / "experiments"

if __name__ == "__main__":
    out = {}
    boards = (("random", SoccerGame(move_order="random")), ("deterministic", A10SoccerGame()))
    for board, game in boards:
        solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
        _, row_t, col_t = fh.solve_finite_horizon(solver, 0.9, game.max_steps)
        kick = game.initial_state()
        out[board] = {"state": list(kick),
                      "row": [round(float(v), 6) for v in fh.tables_to_policy(row_t)(0, [kick])[0]],
                      "col": [round(float(v), 6) for v in fh.tables_to_policy(col_t)(0, [kick])[0]]}
        print(board, out[board], flush=True)
    (EXP / "pg_exact_kickoff.json").write_text(json.dumps(out) + "\n")
