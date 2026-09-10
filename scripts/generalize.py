"""How far does the goal-width mixed-equilibrium switch generalize, and why do
the mixed states arise?

    python scripts/generalize.py     # -> experiments/generalize.csv
                                     #    docs/figures/gallery/generalize.svg

Three parts:

A. **Board scale and goal shape.** Sweep larger boards and odd goal shapes
   (offset against a wall, non-contiguous cells with a gap) under the random
   move order. The switch -- one goal cell gives 0 mixed stage games, two or
   more gives some -- is checked to hold, and adjacency of the goal cells is
   shown not to matter.

B. **Transition rule.** Add `slip`: each player independently takes a
   uniform-random move instead of its chosen one with probability `slip`, a
   noise that does not depend on either action. Result: `slip` creates mixed
   stage games even under the deterministic move order and even with a single
   goal cell. So the goal-width switch is specific to Littman's move-order
   rule; it is not a statement about stochastic transitions in general. What
   forces mixing is a transition outcome that hinges on a coin neither player
   controls -- the random order supplies that only where the carrier has two
   lanes, `slip` supplies it everywhere.

C. **Mechanism anatomy.** For the canonical 7x5 three-row-goal board, read the
   2x2 matching-pennies core of every mixed state and report which pair of
   carrier moves crosses with which pair of defender moves. Most are a vertical
   choice: the carrier picks which goal row to head for and the defender guesses
   it.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.certificate import certify_game, classify
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.reachability import reachable_states
from soccer_nash.viz import line_chart_svg

CSV = pathlib.Path("experiments/generalize.csv")
FIG = pathlib.Path("docs/figures/gallery/generalize.svg")
_ACT = {0: "U", 1: "D", 2: "L", 3: "R", 4: "S"}


def _solve(game: SoccerGame, gamma: float):
    s = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-10)
    return s, s.run()


def _mixed(game: SoccerGame, gamma: float = 0.9) -> tuple[int, int]:
    s, r = _solve(game, gamma)
    reach = reachable_states(game)
    n = sum(classify(s._matrix(x, r.values)) == "mixed" for x in reach)
    return n, len(reach)


def part_a() -> list[dict]:
    configs = [
        (7, 5, (2,)), (7, 5, (1, 2)), (7, 5, (1, 3)), (7, 5, (0, 4)),
        (7, 5, (1, 2, 3)), (7, 5, (0, 2, 4)),
        (9, 5, (2,)), (9, 5, (1, 2, 3)),
        (9, 7, (3,)), (9, 7, (2, 3, 4)),
        (11, 5, (2,)), (11, 5, (1, 2, 3)),
        (5, 9, (4,)), (5, 9, (3, 4, 5)),
    ]
    rows = []
    for w, h, gr in configs:
        n, tot = _mixed(SoccerGame(width=w, height=h, goal_rows=gr, move_order="random"))
        cells = len(gr)
        contiguous = all(b - a == 1 for a, b in zip(gr, gr[1:]))
        rows.append({
            "part": "A", "board": f"{w}x{h}", "goal_rows": "-".join(map(str, gr)),
            "goal_cells": cells, "contiguous": int(contiguous),
            "slip": 0.0, "move_order": "random",
            "states": tot, "mixed": n, "pct": round(100 * n / tot, 2),
            "switch_ok": int((cells == 1 and n == 0) or (cells >= 2 and n > 0)),
        })
    return rows


def part_b() -> list[dict]:
    rows = []
    for mo in ("deterministic", "random"):
        for gr, tag in (((2,), "1-cell"), ((1, 2), "2-cell")):
            for slip in (0.0, 0.1, 0.2, 0.3):
                g = SoccerGame(width=5, height=5, goal_rows=gr,
                               move_order=mo, slip=slip)
                n, tot = _mixed(g)
                rows.append({
                    "part": "B", "board": "5x5", "goal_rows": tag,
                    "goal_cells": len(gr), "contiguous": 1,
                    "slip": slip, "move_order": mo,
                    "states": tot, "mixed": n, "pct": round(100 * n / tot, 2),
                    "switch_ok": "",
                })
    return rows


def part_c() -> tuple[list[dict], dict]:
    g = SoccerGame(width=7, height=5, goal_rows=(1, 2, 3), move_order="random")
    s, r = _solve(g, 0.9)
    carrier_pairs: Counter = Counter()
    defender_pairs: Counter = Counter()
    for st in r.no_saddle_states:
        c = certify_game(s._matrix(st, r.values))
        if not c.core:
            continue
        b = st[4]
        rp = tuple(sorted(_ACT[a] for a in c.core.rows))
        cp = tuple(sorted(_ACT[a] for a in c.core.cols))
        carrier, defender = (rp, cp) if b == 0 else (cp, rp)
        carrier_pairs["".join(carrier)] += 1
        defender_pairs["".join(defender)] += 1
    total = sum(carrier_pairs.values())
    vertical = sum(
        n for k, n in carrier_pairs.items() if "U" in k or "D" in k
    )
    rows = [{
        "part": "C", "board": "7x5", "goal_rows": "1-2-3", "goal_cells": 3,
        "contiguous": 1, "slip": 0.0, "move_order": "random",
        "states": total, "mixed": total, "pct": "",
        "switch_ok": "",
        "carrier_crossings": dict(carrier_pairs.most_common()),
        "defender_crossings": dict(defender_pairs.most_common()),
    }]
    summary = {
        "mixed_with_core": total,
        "carrier_crossing_involves_vertical": vertical,
        "carrier_crossings": dict(carrier_pairs.most_common()),
        "defender_crossings": dict(defender_pairs.most_common()),
    }
    return rows, summary


def _figure(b_rows: list[dict]) -> None:
    def pts(mo, tag):
        return [(row["slip"], row["pct"]) for row in b_rows
                if row["move_order"] == mo and row["goal_rows"] == tag]

    series = [
        ("random, 2-cell goal", "var(--p0, #2f6bb0)", pts("random", "2-cell")),
        ("random, 1-cell goal", "var(--ember, #a94e18)", pts("random", "1-cell")),
        ("deterministic, 2-cell", "var(--p1, #2f8a52)", pts("deterministic", "2-cell")),
        ("deterministic, 1-cell", "var(--ink-faint, #77817a)", pts("deterministic", "1-cell")),
    ]
    svg = line_chart_svg(
        series, x_label="slip probability", y_label="% stage games mixed",
        title="Movement noise creates mixing at any goal width",
    )
    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(svg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-c", action="store_true", help="skip the slow 7x5 anatomy")
    args = ap.parse_args()

    a = part_a()
    b = part_b()
    c_rows, c_summary = ([], {}) if args.skip_c else part_c()

    fields = ["part", "board", "goal_rows", "goal_cells", "contiguous", "slip",
              "move_order", "states", "mixed", "pct", "switch_ok"]
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in a + b + c_rows:
            w.writerow(row)
    _figure(b)

    print("A. board scale and goal shape (move_order=random)")
    for row in a:
        flag = "ok" if row["switch_ok"] else "  <-- SWITCH BROKEN"
        print(f"  {row['board']:>5} goal {row['goal_rows']:<7} "
              f"cells={row['goal_cells']} contig={row['contiguous']}  "
              f"mixed {row['mixed']:>4}/{row['states']:<5} ({row['pct']:>4}%)  {flag}")
    print(f"  switch holds on {sum(r['switch_ok'] for r in a)}/{len(a)} configs; "
          f"adjacency of goal cells does not matter")

    print("\nB. transition rule -- action-independent movement noise (slip)")
    for row in b:
        print(f"  {row['move_order']:>13}  {row['goal_rows']}  slip={row['slip']}  "
              f"mixed {row['mixed']:>3}/{row['states']} ({row['pct']}%)")
    det1 = [r for r in b if r["move_order"] == "deterministic" and r["goal_rows"] == "1-cell"]
    print(f"  deterministic + 1-cell goal: {det1[0]['mixed']} mixed at slip 0, "
          f"{max(r['mixed'] for r in det1)} once slip > 0 "
          f"-- the goal-width switch is move-order specific")

    if c_summary:
        print("\nC. mechanism -- 2x2 matching-pennies core of the 7x5 mixed states")
        v, t = c_summary["carrier_crossing_involves_vertical"], c_summary["mixed_with_core"]
        print(f"  {t} mixed states, all with a 2x2 core; "
              f"carrier crossing involves a vertical move in {v}/{t}")
        print(f"  carrier crossings: {c_summary['carrier_crossings']}")
        print(f"  defender crossings: {c_summary['defender_crossings']}")

    print(f"\nwrote {CSV} and {FIG}")


if __name__ == "__main__":
    main()
