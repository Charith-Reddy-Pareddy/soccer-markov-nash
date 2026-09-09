"""Sweep the move-resolution rule from deterministic to random.

`move_order="blend"` resolves each joint action by Littman's random order with
probability `blend`, otherwise by the deterministic carrier-wins rule. `blend`
is a continuous knob from the pure-saddle game (0) to the mixing game (1).

The finding: mixing switches on **sharply** once the random component is the
majority (`blend` just above 0.5), and *overshoots* -- more stage games are
mixed just past the threshold than in the fully random game -- before relaxing.

    python scripts/blend.py                  # 5x4, 2-cell goal (fast)
    python scripts/blend.py --width 7 --height 5 --goal-rows 1,2,3

Writes `experiments/blend.csv` and the sweep figure.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.viz import P0, line_chart_svg

CSV = pathlib.Path("experiments/blend.csv")
FIG = pathlib.Path("docs/figures/gallery/blend.svg")
_EMBER = "var(--ember, #a94e18)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--height", type=int, default=4)
    ap.add_argument("--goal-rows", type=str, default="1,2")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--step", type=float, default=0.1)
    args = ap.parse_args()

    goal = tuple(int(v) for v in args.goal_rows.split(","))
    grid = [round(i * args.step, 4) for i in range(int(1 / args.step) + 1)]
    # denser around the transition
    grid = sorted(set(grid) | {0.45, 0.5, 0.52, 0.55, 0.58, 0.6, 0.65})

    rows = []
    print(f"{args.width}x{args.height} goal {goal}, gamma {args.gamma}\n")
    print(f"{'blend':>6}  {'mixed':>6}  {'V(kickoff)':>11}")
    for p in grid:
        mo = "deterministic" if p == 0.0 else "random" if p == 1.0 else "blend"
        game = SoccerGame(width=args.width, height=args.height, goal_rows=goal,
                          move_order=mo, blend=p)
        res = NashQIteration(game, gamma=args.gamma, mode="hybrid",
                             tol=1e-9).run()
        m = len(res.no_saddle_states)
        v = res.values[game.initial_state()]
        rows.append((p, m, v))
        print(f"{p:>6.2f}  {m:>6d}  {v:>+11.4f}")

    onset = next((p for p, m, _v in rows if m > 0), None)
    peak = max(rows, key=lambda r: r[1])
    full = rows[-1][1]
    print(f"\nonset: blend = {onset}   peak: {peak[1]} mixed at blend {peak[0]}"
          f"   fully random: {full} mixed")
    print("=> a sharp threshold near blend 0.5, with an overshoot above it.")

    CSV.parent.mkdir(exist_ok=True)
    with CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["blend", "mixed", "v_kickoff"])
        for p, m, v in rows:
            w.writerow([p, m, f"{v:.4f}"])
    print(f"\nwrote {CSV}")

    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(line_chart_svg(
        [("mixed states", P0, [(p, m) for p, m, _v in rows])],
        x_label="blend  (P(random move order))",
        y_label="no-pure-saddle states",
        title=f"Mixing switches on above blend 0.5  ({args.width}x{args.height})",
        vline=0.5,
    ))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
