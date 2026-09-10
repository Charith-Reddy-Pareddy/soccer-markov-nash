"""The ``tackle`` collision rule: this project's own transition design.

    python scripts/tackle.py     # -> experiments/tackle.csv
                                 #    docs/figures/gallery/rule_fingerprints.svg
                                 #    docs/figures/gallery/tackle_sweep.svg

The rule (``soccer_nash/game.py``): the defender commits to a challenge by
moving onto the carrier's cell. The challenge is a duel -- it wins the ball with
probability ``tackle_prob`` and otherwise fails and bounces the defender back.
"Dive in or contain" is a real gamble, so the rule has its own mixed-strategy
region.

Findings:

* Like ``slip`` and unlike Littman's random move order, ``tackle`` produces
  mixed stage games **at any goal width**, single cell included. It is the duel,
  not the goal geometry, that forces the guess -- another confirmation that the
  goal-width switch is specific to the move-order rule.
* The tackle-mixed states sit in the "dive-in zone": the defender within a move
  of the carrier, where a committed challenge is on the table.
* The rule-fingerprint panel shows each collision rule's mixed region as a
  distinct shape on the same board.
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
from soccer_nash.geometry import features
from soccer_nash.nash_q import NashQIteration
from soccer_nash.reachability import reachable_states
from soccer_nash.viz import bestreply_svg, line_chart_svg, mixing_map_svg, panel_svg

CSV = pathlib.Path("experiments/tackle.csv")
FINGERPRINTS = pathlib.Path("docs/figures/gallery/rule_fingerprints.svg")
SWEEP = pathlib.Path("docs/figures/gallery/tackle_sweep.svg")

# (label, move_order, extra kwargs)
RULES = [
    ("deterministic", "deterministic", {}),
    ("coinflip", "coinflip", {}),
    ("random", "random", {}),
    ("blend 0.7", "blend", {"blend": 0.7}),
    ("slip 0.2", "deterministic", {"slip": 0.2}),
    ("tackle 0.5", "tackle", {"tackle_prob": 0.5}),
]


def _solve(game: SoccerGame):
    s = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    return s, s.run()


def _mixed(game: SoccerGame) -> tuple[int, int, object, object]:
    s, r = _solve(game)
    reach = reachable_states(game)
    n = sum(classify(s._matrix(x, r.values)) == "mixed" for x in reach)
    return n, len(reach), s, r


def fingerprints() -> list[str]:
    """One mixing map per collision rule, defender pinned at the goal mouth."""
    panels = []
    for label, mo, kw in RULES:
        g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order=mo, **kw)
        s, r = _solve(g)
        panels.append(mixing_map_svg(
            g, lambda st, s=s, r=r: s._matrix(st, r.values),
            defender_cell=(4, 1), defender=1, title=label, caption=None,
        ))
    return panels


def sweep() -> list[dict]:
    rows = []
    for gr, tag in (((2,), "1-cell"), ((1, 2), "2-cell")):
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9):
            g = SoccerGame(width=5, height=4, goal_rows=gr,
                           move_order="tackle", tackle_prob=q)
            n, tot, _s, _r = _mixed(g)
            rows.append({"goal": tag, "tackle_prob": q,
                         "states": tot, "mixed": n,
                         "pct": round(100 * n / tot, 2)})
    return rows


def anatomy() -> tuple[dict, str]:
    """Where the tackle-mixed states sit, and a close-up of one."""
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="tackle",
                   tackle_prob=0.5)
    s, r = _solve(g)
    dists: Counter = Counter()
    example = None
    for st in r.no_saddle_states:
        f = features(g, st)
        dists[f["player_dist"]] += 1
        if example is None and f["player_dist"] == 1:
            example = st
    close = ""
    if example is not None:
        m = s._matrix(example, r.values)
        cert = certify_game(m)
        if cert.core:
            import numpy as np
            A = np.array(cert.core.submatrix)
            b = example[4]
            rl = [f"carrier {a}" for a in cert.core.rows] if b == 0 else \
                 [f"defender {a}" for a in cert.core.rows]
            cl = [f"defender {a}" for a in cert.core.cols] if b == 0 else \
                 [f"carrier {a}" for a in cert.core.cols]
            close = bestreply_svg(A, rl, cl,
                                  title="Dive in or contain: the tackle guess")
    return {"player_dist_of_mixed": dict(sorted(dists.items())),
            "example_state": list(example) if example else None}, close


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    print("mixed stage games by collision rule (5x4, goal (1,2), defender free)")
    for label, mo, kw in RULES:
        g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order=mo, **kw)
        n, tot, _s, _r = _mixed(g)
        print(f"  {label:<14} {n:>3}/{tot}")

    swp = sweep()
    print("\ntackle_prob sweep (5x4)")
    for row in swp:
        print(f"  {row['goal']}  q={row['tackle_prob']:<4} "
              f"mixed {row['mixed']:>3}/{row['states']} ({row['pct']}%)")
    one = [r for r in swp if r["goal"] == "1-cell"]
    print(f"  1-cell goal + tackle: {one[0]['mixed']} mixed at q=0, "
          f"{max(r['mixed'] for r in one)} once q>0 -- mixing at any goal width")

    anat, close = anatomy()
    print(f"\ntackle-mixed states by carrier/defender distance: "
          f"{anat['player_dist_of_mixed']}")
    print("  -> concentrated where a committed challenge is reachable")

    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["goal", "tackle_prob", "states", "mixed", "pct"])
        w.writeheader()
        w.writerows(swp)

    FINGERPRINTS.parent.mkdir(parents=True, exist_ok=True)
    FINGERPRINTS.write_text(panel_svg(fingerprints(), cols=3, gap=18))

    series = [
        ("2-cell goal", "var(--p0, #2f6bb0)",
         [(r["tackle_prob"], r["pct"]) for r in swp if r["goal"] == "2-cell"]),
        ("1-cell goal", "var(--ember, #a94e18)",
         [(r["tackle_prob"], r["pct"]) for r in swp if r["goal"] == "1-cell"]),
    ]
    SWEEP.write_text(line_chart_svg(
        series, x_label="tackle_prob", y_label="% stage games mixed",
        title="The tackle duel forces mixing at any goal width",
    ))
    if close:
        pathlib.Path("docs/figures/gallery/tackle_core.svg").write_text(close)

    print(f"\nwrote {CSV}, {FINGERPRINTS}, {SWEEP}")


if __name__ == "__main__":
    main()
