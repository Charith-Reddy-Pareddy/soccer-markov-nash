"""One function that solves a game configuration and returns a flat result row.

Every experiment script writes rows from :func:`run_config` to a CSV under
``experiments/`` so the report's tables and figures regenerate from data.
"""

from __future__ import annotations

import csv
import pathlib
import time
from typing import Iterable

from soccer_nash.exploit import duality_gap
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.numerics import classify_stage_game

FIELDS = [
    "move_order",
    "gamma",
    "width",
    "height",
    "goal_rows",
    "rel_tol",
    "states",
    "pure_states",
    "degenerate_states",
    "mixed_states",
    "mixed_fraction",
    "iterations",
    "lp_calls",
    "runtime_s",
    "kickoff_value",
    "max_value_bracket_gap",
    "duality_gap",
]


def standard_goal_rows(height: int) -> tuple[int, ...]:
    """The goal mouth: every row except the top and bottom (a single middle row
    for very short boards)."""
    if height <= 3:
        return (height // 2,)
    return tuple(range(1, height - 1))


def run_config(
    move_order: str = "deterministic",
    gamma: float = 0.9,
    width: int = 7,
    height: int = 5,
    goal_rows: tuple[int, ...] = (1, 2, 3),
    rel_tol: float = 1e-6,
    tol: float = 1e-9,
    measure_exploitability: bool = False,
) -> dict:
    game = SoccerGame(
        width=width, height=height, goal_rows=goal_rows, move_order=move_order
    )
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=tol)

    start = time.perf_counter()
    result = solver.run()
    runtime = time.perf_counter() - start

    counts = {"pure": 0, "degenerate": 0, "mixed": 0}
    for s in solver._states:
        counts[classify_stage_game(solver._matrix(s, result.values), rel_tol)] += 1
    n = len(solver._states)

    gap = float("nan")
    if measure_exploitability:
        gap = duality_gap(game, result.row_policy, result.col_policy, gamma=gamma)

    return {
        "move_order": move_order,
        "gamma": gamma,
        "width": width,
        "height": height,
        "goal_rows": "|".join(map(str, goal_rows)),
        "rel_tol": rel_tol,
        "states": n,
        "pure_states": counts["pure"],
        "degenerate_states": counts["degenerate"],
        "mixed_states": counts["mixed"],
        "mixed_fraction": counts["mixed"] / n if n else 0.0,
        "iterations": result.iterations,
        "lp_calls": result.matrix_game_solves,
        "runtime_s": round(runtime, 3),
        "kickoff_value": round(result.values[game.initial_state()], 6),
        "max_value_bracket_gap": float(solver.value_bracket_gaps(result).max()),
        "duality_gap": gap,
    }


def write_csv(path: str | pathlib.Path, rows: Iterable[dict]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
