"""Solve the deterministic-transition soccer game with a dense possession +
progression reward (winning +1, losing -1, possession 0.005, progression
0.005*x0, gamma 0.9) and print the equilibrium policy at kickoff.

This legacy pre-state experiment is not Jae's current reward formula.
Use scripts/reward_q_audit.py for the verified next-state comparison.

    python scripts/possession_progression_policy.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.shaping import CombinedShaping, StepPossessionBonus, StepProgressionBonus

g = SoccerGame(width=7, height=5, goal_rows=(1, 2, 3), move_order="deterministic")
shaping = CombinedShaping(StepPossessionBonus(0.005), StepProgressionBonus(0.005))
result = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10, shaping=shaping).run()

s = g.initial_state()
print(f"state: {s}")
print(f"row_policy (player 0): {result.row_policy[s].round(4).tolist()}")
print(f"col_policy (player 1): {result.col_policy[s].round(4).tolist()}")
