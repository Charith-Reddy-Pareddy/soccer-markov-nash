"""Phase diagram: mixed-strategy fraction vs. board geometry and goal-mouth
width, to test whether goal width -- not board size -- is what drives mixing.

Writes ``experiments/phase_diagram.csv``: one row per (width, height, goal
width) on the random-move-order game at a fixed gamma.

    python scripts/phase_diagram.py            # ~8 min, writes the CSV
    python scripts/phase_diagram.py --quick    # small boards only
    python scripts/phase_diagram.py --svg-only # just redraw the heatmap from the CSV
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.numerics import classify_stage_game
from soccer_nash.reachability import reachable_states

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "phase_diagram.csv"
SVG = ROOT / "docs" / "figures" / "phase_diagram.svg"
FIELDS = [
    "width", "height", "goal_width", "gamma",
    "states", "reachable", "mixed_states", "mixed_fraction",
    "mixed_fraction_reachable", "iterations", "lp_calls", "runtime_s",
]


def centered_goal_rows(height: int, k: int) -> tuple[int, ...]:
    start = (height - k) // 2
    return tuple(range(start, start + k))


def run_one(width: int, height: int, k: int, gamma: float) -> dict:
    goal_rows = centered_goal_rows(height, k)
    game = SoccerGame(width=width, height=height, goal_rows=goal_rows,
                      move_order="random")
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-9)

    t = time.perf_counter()
    result = solver.run()
    runtime = time.perf_counter() - t

    reach = reachable_states(game)
    mixed = 0
    mixed_reach = 0
    for s in solver._states:
        if classify_stage_game(solver._matrix(s, result.values)) == "mixed":
            mixed += 1
            if s in reach:
                mixed_reach += 1
    n = len(solver._states)
    return {
        "width": width,
        "height": height,
        "goal_width": k,
        "gamma": gamma,
        "states": n,
        "reachable": len(reach),
        "mixed_states": mixed,
        "mixed_fraction": round(mixed / n, 6),
        "mixed_fraction_reachable": round(mixed_reach / len(reach), 6),
        "iterations": result.iterations,
        "lp_calls": result.matrix_game_solves,
        "runtime_s": round(runtime, 2),
    }


def configs(quick: bool) -> list[tuple[int, int]]:
    widths = (3, 5, 7) if quick else (3, 5, 7, 9)
    cap = 30 if quick else 45
    out = []
    for w in widths:
        for h in range(3, 8):
            if w * h <= cap:
                out.append((w, h))
    return out


def _heat(frac: float, hi: float) -> str:
    """A pale-to-ember fill for a mixed fraction in ``[0, hi]``."""
    if frac <= 0:
        return "#eef2ec"
    p = 0.12 + min(frac / hi, 1.0) * 0.78  # share of ember vs white
    ember = (169, 78, 24)
    r, g, b = (round(c * p + 255 * (1 - p)) for c in ember)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_heatmap(rows: list[dict], path: pathlib.Path) -> None:
    boards = sorted({(r["width"], r["height"]) for r in rows})
    ks = sorted({r["goal_width"] for r in rows})
    frac = {(r["width"], r["height"], r["goal_width"]): r["mixed_fraction"] for r in rows}
    hi = max(frac.values())

    cell, left, top = 40, 66, 46
    W = max(left + len(ks) * cell + 20, 300)
    H = top + len(boards) * cell + 34
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Mixed-strategy fraction by board size (rows) and goal-mouth '
        f'width (columns). The goal-width-1 column is zero for every board.">',
        '<style>text{font-family:ui-monospace,monospace;font-size:11px;'
        'fill:var(--ink-faint,#77817a)}</style>',
    ]
    for j, k in enumerate(ks):
        out.append(f'<text x="{left + j * cell + cell / 2:.0f}" y="{top - 14}" '
                   f'text-anchor="middle">{k}</text>')
    out.append(f'<text x="{left + len(ks) * cell / 2:.0f}" y="{top - 30}" '
               f'text-anchor="middle">goal-mouth width</text>')
    for i, (w, h) in enumerate(boards):
        y = top + i * cell
        out.append(f'<text x="{left - 8}" y="{y + cell / 2 + 4:.0f}" '
                   f'text-anchor="end">{w}x{h}</text>')
        for j, k in enumerate(ks):
            x = left + j * cell
            f = frac.get((w, h, k))
            if f is None:
                continue
            out.append(f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" '
                       f'rx="2" fill="{_heat(f, hi)}" stroke="var(--rule,#d9e2db)"/>')
            label = "0" if f == 0 else f"{f * 100:.0f}"
            out.append(f'<text x="{x + cell / 2 - 1:.0f}" y="{y + cell / 2 + 4:.0f}" '
                       f'text-anchor="middle" fill="var(--ink,#19211c)">{label}</text>')
    out.append(f'<text x="{left}" y="{H - 12}">cells = mixed-state %</text>')
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))


def load_rows() -> list[dict]:
    with OUT.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("width", "height", "goal_width", "states", "reachable",
                    "mixed_states", "iterations", "lp_calls"):
            r[key] = int(r[key])
        for key in ("gamma", "mixed_fraction", "mixed_fraction_reachable", "runtime_s"):
            r[key] = float(r[key])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--svg-only", action="store_true",
                        help="redraw docs/figures/phase_diagram.svg from the CSV")
    args = parser.parse_args()

    if args.svg_only:
        write_heatmap(load_rows(), SVG)
        print(f"wrote {SVG.relative_to(ROOT)}")
        return

    rows = []
    for w, h in configs(args.quick):
        for k in range(1, h - 1):  # goal width 1 .. height-2
            row = run_one(w, h, k, args.gamma)
            rows.append(row)
            print(f"  {w}x{h} goal {k}: mixed {row['mixed_states']:4d}  "
                  f"({row['mixed_fraction']:.3f})  {row['runtime_s']:.1f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_heatmap(rows, SVG)
    print(f"wrote {OUT.relative_to(ROOT)} and {SVG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
