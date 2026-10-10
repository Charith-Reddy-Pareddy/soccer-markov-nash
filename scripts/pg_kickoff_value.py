"""Value of the random move-order board at the kickoff, for the exact 100-step discounted
solution (what the ball-holding player is worth even against a perfect best response).

    python scripts/pg_kickoff_value.py     # experiments/pg_random_kickoff_value.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash import finite_horizon as fh
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration

EXP = pathlib.Path(__file__).resolve().parent.parent / "experiments"

if __name__ == "__main__":
    game = SoccerGame(move_order="random")
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    values, _, _ = fh.solve_finite_horizon(solver, 0.9, game.max_steps)
    out = {"state": list(game.initial_state()), "value": round(values[0][game.initial_state()], 6)}
    (EXP / "pg_random_kickoff_value.json").write_text(json.dumps(out) + "\n")
    print(out)
