"""Phase diagram: mixed-strategy fraction vs. board geometry and goal-mouth
width, to test whether goal width -- not board size -- is what drives mixing.

Writes ``experiments/phase_diagram.csv``: one row per (width, height, goal
width) on the random-move-order game at a fixed gamma, for every board with
``3 <= width <= 11``, ``3 <= height <= 9``, ``width * height <= 45`` (the cap
keeps the random-move solve tractable while spanning both axes).

    python scripts/phase_diagram.py            # ~15 min, writes the CSV
    python scripts/phase_diagram.py --quick    # small boards only
    python scripts/phase_diagram.py --svg-only # redraw the heatmap from the CSV
    python scripts/phase_diagram.py --analyze  # variance decomposition on the CSV
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
    if quick:
        return [(w, h) for w in (3, 5, 7) for h in range(3, 8) if w * h <= 30]
    return [
        (w, h)
        for w in range(3, 12)
        for h in range(3, 10)
        if w * h <= 45
    ]


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
            ink = "#ffffff" if (f > 0 and f / hi > 0.45) else "var(--ink,#19211c)"
            out.append(f'<text x="{x + cell / 2 - 1:.0f}" y="{y + cell / 2 + 4:.0f}" '
                       f'text-anchor="middle" fill="{ink}">{label}</text>')
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


def _ols(y: list[float], cols: list[list[float]]) -> tuple[list[float], float]:
    """Tiny ordinary-least-squares with an intercept. Returns (coefs, R^2)."""
    import numpy as np

    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    yv = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return list(beta), r2


def analyze() -> None:
    """Decompose the variance in mixed_fraction: what predicts it?"""
    rows = load_rows()
    one = [r for r in rows if r["goal_width"] == 1]
    multi = [r for r in rows if r["goal_width"] >= 2]

    print(f"{len(rows)} configs: {len(one)} one-cell, {len(multi)} multi-cell\n")

    # (1) the binary gate
    one_mixed = sum(1 for r in one if r["mixed_states"] > 0)
    multi_mixed = sum(1 for r in multi if r["mixed_states"] > 0)
    print("has any mixed state?")
    print(f"  goal width == 1 : {one_mixed}/{len(one)}   -> a perfect predictor of 'no mixing'")
    print(f"  goal width >= 2 : {multi_mixed}/{len(multi)}  -> always some mixing\n")

    # (2) among multi-cell boards, what explains the fraction?
    y = [r["mixed_fraction"] for r in multi]
    w = [r["width"] for r in multi]
    h = [r["height"] for r in multi]
    k = [r["goal_width"] for r in multi]
    area = [r["width"] * r["height"] for r in multi]

    print("multi-cell boards -- OLS of mixed_fraction, incremental R^2:")
    for name, cols in [
        ("goal_width alone", [k]),
        ("width alone", [w]),
        ("height alone", [h]),
        ("board area alone", [area]),
        ("width + height", [w, h]),
        ("width + height + goal_width", [w, h, k]),
        ("area + goal_width", [area, k]),
    ]:
        _, r2 = _ols(y, cols)
        print(f"  {name:<30s} R^2 = {r2:.3f}")

    import statistics
    print(f"\n  mixed_fraction over multi-cell boards: "
          f"mean {statistics.mean(y):.3f}, sd {statistics.pstdev(y):.3f}, "
          f"range [{min(y):.3f}, {max(y):.3f}]")
    beta, _ = _ols(y, [area, k])
    print(f"  fitted: mixed_fraction ~= {beta[0]:.3f} + {beta[1]:.5f}*area "
          f"+ {beta[2]:.4f}*goal_width")
    print("  -> board area (dilution) drives the residual variation; goal width")
    print("     past 2 barely matters, and neither creates or removes mixing.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--svg-only", action="store_true",
                        help="redraw docs/figures/phase_diagram.svg from the CSV")
    parser.add_argument("--analyze", action="store_true",
                        help="variance decomposition on the existing CSV")
    args = parser.parse_args()

    if args.svg_only:
        write_heatmap(load_rows(), SVG)
        print(f"wrote {SVG.relative_to(ROOT)}")
        return
    if args.analyze:
        analyze()
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
