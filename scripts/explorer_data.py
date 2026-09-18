"""Export every board used in docs/positions.pdf, exactly solved, to a
compact JSON file for the interactive board explorer (the React app in
``site/``, built to ``docs/explorer.html``).

For every non-terminal state of each board, writes ``V`` (player 0's exact
value), ``row_policy`` / ``col_policy`` (player 0 / player 1's 4-action
equilibrium mix), and the raw (unoriented) ``Q`` matrix (player 0 = row,
player 1 = col) -- the same ``NashQIteration.run_exact()`` values
docs/positions.pdf is built from, so the site and the PDF never disagree.
Reorientation (carrier = row), wall-clamp ("hold") detection, and the
best-response node-and-arrow graph are cheap enough to do client-side, so
they are not precomputed here.

    python scripts/explorer_data.py

Writes ``docs/data/explorer.json``, fetched at runtime by the explorer page.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration

OUT = pathlib.Path("docs/data/explorer.json")

# Every board any case in docs/positions.md is solved on.
BOARDS: dict[str, dict] = {
    "canonical": {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "random"},
    "tackle": {
        "width": 5, "height": 4, "goal_rows": (1, 2),
        "move_order": "tackle", "tackle_prob": 0.5,
    },
    "territory": {
        "width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "deterministic",
        "scoring": "territory", "territory_reward": 0.05,
    },
    "slip": {
        "width": 5, "height": 5, "goal_rows": (2,),
        "move_order": "deterministic", "slip": 0.15,
    },
}
BOARD_LABELS = {
    "canonical": "Canonical board — random move order",
    "tackle": "Tackle rule",
    "territory": "Territory reward, deterministic",
    "slip": "Movement slip, deterministic",
}


def r4(x: float) -> float:
    return round(float(x), 4)


def solve_board(bid: str, kw: dict) -> dict:
    g = SoccerGame(**kw)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run_exact()
    print(f"  {bid}: |V_exact - V_iterative| = {result.exact_vs_iterative:.2e} "
          f"over {len(result.values)} states")

    states = {}
    for s in solver._states:
        x0, y0, x1, y1, b = s
        M = solver._matrix(s, result.values)
        key = f"{x0},{y0},{x1},{y1},{b}"
        states[key] = [
            r4(result.values[s]),
            [r4(p) for p in result.row_policy[s]],
            [r4(p) for p in result.col_policy[s]],
            [[r4(M[i, j]) for j in range(4)] for i in range(4)],
        ]

    return {
        "label": BOARD_LABELS[bid],
        "width": g.width,
        "height": g.height,
        "goal_rows": list(g.goal_rows),
        "exact_vs_iterative": result.exact_vs_iterative,
        "no_saddle_count": len(result.no_saddle_states),
        "state_count": len(states),
        "states": states,
    }


def main() -> None:
    print("exact solve, one board at a time:")
    boards = {bid: solve_board(bid, kw) for bid, kw in BOARDS.items()}

    payload = {"gamma": 0.9, "boards": boards}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    total_states = sum(b["state_count"] for b in boards.values())
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB, "
          f"{len(boards)} boards, {total_states} states total)")


if __name__ == "__main__":
    main()
