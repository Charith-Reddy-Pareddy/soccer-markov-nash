"""Export the exact-solved canonical board to a compact JS data file for the
interactive board explorer (docs/explorer.html).

For every non-terminal state, writes ``V`` (player 0's exact value),
``row_policy`` / ``col_policy`` (player 0 / player 1's 4-action equilibrium
mix), and the raw (unoriented) ``Q`` matrix (player 0 = row, player 1 =
col) -- the same ``NashQIteration.run_exact()`` values docs/positions.pdf
is built from, so the site and the PDF never disagree. Reorientation
(carrier = row) and wall-clamp ("hold") detection are cheap enough to do
client-side, so they are not precomputed here.

    python scripts/explorer_data.py

Writes ``docs/data/explorer.js`` (a single ``window.EXPLORER = {...}``
assignment, not JSON-over-fetch, so the page works from a plain ``file://``
open with no local server and no CORS friction).
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration

OUT = pathlib.Path("docs/data/explorer.js")
CANON = {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "random"}


def r4(x: float) -> float:
    return round(float(x), 4)


def main() -> None:
    g = SoccerGame(**CANON)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run_exact()
    print(f"exact solve: |V_exact - V_iterative| = {result.exact_vs_iterative:.2e} "
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

    payload = {
        "width": g.width,
        "height": g.height,
        "goal_rows": list(g.goal_rows),
        "gamma": solver.gamma,
        "exact_vs_iterative": result.exact_vs_iterative,
        "no_saddle_count": len(result.no_saddle_states),
        "state_count": len(states),
        "states": states,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = "window.EXPLORER = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    OUT.write_text(text)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB, {len(states)} states)")


if __name__ == "__main__":
    main()
