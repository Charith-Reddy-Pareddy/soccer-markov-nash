"""Classify the 94 no-pure-saddle states, not just count them.

"94 mixed states" conflates three different things, per the concern that
some of `positions.md`'s cases (4, 5, 9, 10) turned out to have a zero-weight
action tying the LP-reported support's value exactly -- the reported split
is *a* valid equilibrium, not necessarily the only one:

- a state has **no pure saddle** iff its value-bracket gap is nonzero
  (`certify_game`) -- this is what the "94" count actually measures;
- the LP returns some equilibrium `(p, q)` for it, which may have
  **fractional support** on one or both sides;
- whether that fractional support is **strategically necessary** -- forced
  by keeping the opponent indifferent (a genuine matching-pennies core) --
  or merely **degenerate** -- an action outside the reported support ties
  its value exactly, so the reported split is one point on a larger face of
  equally-valid equilibria, not a forced number -- is a separate question
  this script answers directly, per state.

For each of the 94 no-pure-saddle states on the canonical board, checks
every action *not* in the LP-reported support: does its expected payoff
against the opponent's actual strategy tie the support's common value
(`tol=1e-6`, the same tolerance `support_shape` and `scripts/positions.py`'s
`_support` already use)? Classifies each state as:

- **unique** -- neither side has a tied zero-weight action; the reported
  support is the whole indifference class, forced by the matching-pennies
  structure with no freedom left over.
- **degenerate equilibrium face** -- a side whose reported support already
  has >= 2 actions also has a zero-weight action tied with it (Cases 4 and
  5): the LP found one vertex of a larger polytope face, not a uniquely
  forced split.
- **pure reply tied inside the equilibrium set** -- a side whose reported
  support is a single action (nominally "pure") has a zero-weight action
  tied with it (Case 9's defender, Case 10's defender): the "pure" reply
  isn't uniquely forced either, it's just the vertex the LP happened to
  report.

A state can show both patterns (one per side); it is counted once under
each that applies, plus once under "unique" only if neither applies.

    python scripts/degeneracy.py

Writes `experiments/degeneracy.csv`.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.certificate import certify_game
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "degeneracy.csv"
CANON = {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "random"}
TOL = 1e-6


def _oriented_matrix(solver, state, values):
    """Rows = carrier's actions, cols = defender's -- same convention as
    `scripts/positions.py`'s `_oriented_matrix`."""
    M = solver._matrix(state, values)
    return M if state[4] == 0 else -M.T


def _support(p: np.ndarray, tol: float = TOL) -> frozenset[int]:
    return frozenset(int(i) for i, v in enumerate(p) if v > tol)


def _has_tied_outsider(support: frozenset[int], values: np.ndarray, tol: float = TOL) -> bool:
    """Is there an action *not* in `support` whose value ties the support's
    common value (to `tol`)? `values[i]` is E[action i] against the
    opponent's actual strategy -- see `_indifference` in positions.py."""
    if not support:
        return False
    v_support = np.mean([values[i] for i in support])
    return any(
        i not in support and abs(values[i] - v_support) <= tol
        for i in range(len(values))
    )


def classify(solver, state, row_policy, col_policy, values) -> dict:
    M = _oriented_matrix(solver, state, values)
    carrier_pol = row_policy[state] if state[4] == 0 else col_policy[state]
    defender_pol = col_policy[state] if state[4] == 0 else row_policy[state]

    e_carrier = M @ defender_pol          # carrier's E[row] vs defender's mix
    e_defender = carrier_pol @ M          # defender's E[col] vs carrier's mix

    c_support = _support(carrier_pol)
    d_support = _support(defender_pol)
    c_degenerate = _has_tied_outsider(c_support, e_carrier)
    d_degenerate = _has_tied_outsider(d_support, e_defender)

    if not c_degenerate and not d_degenerate:
        kind = "unique"
    else:
        parts = []
        for degenerate, support, who in (
            (c_degenerate, c_support, "carrier"), (d_degenerate, d_support, "defender")
        ):
            if not degenerate:
                continue
            parts.append(
                f"{who} pure-tied" if len(support) == 1 else f"{who} face"
            )
        kind = " + ".join(parts)

    return {
        "state": state,
        "carrier_support": len(c_support),
        "defender_support": len(d_support),
        "carrier_degenerate": c_degenerate,
        "defender_degenerate": d_degenerate,
        "kind": kind,
    }


def main() -> None:
    game = SoccerGame(**CANON)
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run()
    no_saddle = sorted(result.no_saddle_states)

    rows = []
    for s in no_saddle:
        M = _oriented_matrix(solver, s, result.values)
        cert = certify_game(M)
        assert cert.kind == "mixed"  # sanity: still exactly the certified set
        rows.append(classify(solver, s, result.row_policy, result.col_policy,
                              result.values))

    n = len(rows)
    unique = sum(1 for r in rows if r["kind"] == "unique")
    faces = sum(1 for r in rows if "face" in r["kind"])
    pure_tied = sum(1 for r in rows if "pure-tied" in r["kind"])
    both_sides = sum(
        1 for r in rows if r["carrier_degenerate"] and r["defender_degenerate"]
    )

    print(f"{n} no-pure-saddle states on the canonical board (7x5, goal_rows="
          f"(1,2,3), random move order, gamma 0.9)\n")
    print(f"  unique mixed equilibrium (no tied outsider on either side): "
          f"{unique} ({100 * unique / n:.1f}%)")
    print(f"  degenerate equilibrium face (support >= 2 side has a tied "
          f"zero-weight action): {faces} ({100 * faces / n:.1f}%)")
    print(f"  pure reply tied inside the equilibrium set (support == 1 side "
          f"has a tied zero-weight action): {pure_tied} "
          f"({100 * pure_tied / n:.1f}%)")
    print(f"  both sides degenerate at once: {both_sides}")
    print()
    print(f"  {'kind':>45}  count")
    from collections import Counter
    for kind, count in sorted(Counter(r["kind"] for r in rows).items(),
                              key=lambda kv: -kv[1]):
        print(f"  {kind:>45}  {count}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "carrier_support", "defender_support",
                    "carrier_degenerate", "defender_degenerate", "kind"])
        for r in rows:
            w.writerow([r["state"], r["carrier_support"], r["defender_support"],
                       r["carrier_degenerate"], r["defender_degenerate"], r["kind"]])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
