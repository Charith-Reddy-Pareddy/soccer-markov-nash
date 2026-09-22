"""Export every board used in docs/positions.pdf, exactly solved, to a
compact JSON file for the interactive board explorer (the React app in
``site/``, built to ``docs/explorer.html``).

For every non-terminal state of each board, writes ``V`` (player 0's exact
value), ``row_policy`` / ``col_policy`` (player 0 / player 1's 4-action
equilibrium mix), the raw (unoriented) ``Q`` matrix (player 0 = row,
player 1 = col), ``trans`` -- the real ``game.transitions()`` outcome for
every one of the 16 joint actions, so the explorer's "simulate N games"
feature samples real transitions instead of a second, hand-rolled transition
engine in JavaScript that could quietly drift from this one -- and
``rounding``: the matrix solved again at 3/2/1-decimal precision
(:func:`soccer_nash.numerics.rounding_diagnostic`), included only when
rounding actually changes the equilibrium's classification or support
(:func:`soccer_nash.numerics.is_rounding_artifact`), same as
docs/positions.md's own diagnostic tables, so a JS reimplementation of the
zero-sum LP solver is never needed client-side. All of it comes from the
same ``NashQIteration.run_exact()`` solve docs/positions.pdf is built from,
so the site and the PDF never disagree. Reorientation (carrier = row),
wall-clamp ("hold") detection, and the best-response node-and-arrow graph
are cheap enough to do client-side, so they are not precomputed here.

    python scripts/explorer_data.py

Writes ``docs/data/explorer.json``, fetched at runtime by the explorer page.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.game import JOINT_ACTIONS, SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.numerics import is_rounding_artifact, rounding_diagnostic

OUT = pathlib.Path("docs/data/explorer.json")

# Every board any case in docs/positions.md is solved on, plus the
# deterministic (A10-style) twin of the canonical board -- same geometry,
# different move-order rule, so the two can be compared side by side (the
# deterministic/stochastic contrast a member's own site demoed).
BOARDS: dict[str, dict] = {
    "canonical": {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "random"},
    "canonical_det": {
        "width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "deterministic",
    },
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
    "canonical_det": "Canonical board — deterministic (A10-style)",
    "tackle": "Tackle rule",
    "territory": "Territory reward, deterministic",
    "slip": "Movement slip, deterministic",
}


def r6(x: float) -> float:
    # 6 places -- matches docs/positions.pdf's own printed precision, so the
    # site and the PDF show the same "exact" Q matrix, not a coarser rounding
    # of it.
    return round(float(x), 6)


_TERM = {0: -1, 1: -2}  # sentinel indices for a terminal (goal) outcome


def _rounding_entry(M) -> list | None:
    """Rounding diagnostic at 3/2/1 decimals, or ``None`` if none of them is
    a genuine artifact (classification or support change) relative to the
    full-precision equilibrium -- see :func:`is_rounding_artifact`. Only the
    ~3% of states where rounding actually does something get an entry, so
    the payload stays small even though every state was checked.
    """
    diag = rounding_diagnostic(M, decimals=(None, 3, 2, 1))
    full = diag[0]
    rest = diag[1:]
    if not any(is_rounding_artifact(full, d) for d in rest):
        return None
    return [
        [d.decimals, bool(d.pure), r6(d.value), [r6(p) for p in d.row], [r6(p) for p in d.col],
         is_rounding_artifact(full, d)]
        for d in rest
    ]


def _outcome_repr(g: SoccerGame, state, a0, a1, index: dict) -> int | list[list]:
    """One joint action's outcome, as compactly as it can be:

    - a single outcome with probability 1 -> just its state *index* (into
      this board's own state list, not a repeated ``"x0,y0,x1,y1,b"`` string)
      -- the overwhelmingly common case (every deterministic-move-order
      action, most tackle/slip actions too;
    - otherwise -> ``[[index_or_sentinel, prob], ...]``, one pair per
      outcome. ``-1`` / ``-2`` are sentinels for "the game ends here, player
      0 / 1 scored" -- a scored state is terminal and so is never itself a
      key in ``states``.
    """
    outcomes = g.transitions(state, a0, a1)
    if len(outcomes) == 1:
        _prob, ns, _reward = outcomes[0]
        return _TERM[ns[4]] if g.is_terminal(ns) else index[ns]
    out = []
    for prob, ns, _reward in outcomes:
        idx = _TERM[ns[4]] if g.is_terminal(ns) else index[ns]
        out.append([idx, r6(prob)])
    return out


def solve_board(bid: str, kw: dict) -> dict:
    g = SoccerGame(**kw)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run_exact()
    print(f"  {bid}: |V_exact - V_iterative| = {result.exact_vs_iterative:.2e} "
          f"over {len(result.values)} states")

    state_list = list(solver._states)
    index = {s: i for i, s in enumerate(state_list)}

    states = {}
    rounding_flagged = 0
    for i, s in enumerate(state_list):
        x0, y0, x1, y1, b = s
        M = solver._matrix(s, result.values)
        key = f"{x0},{y0},{x1},{y1},{b}"
        rounding = _rounding_entry(M)
        if rounding is not None:
            rounding_flagged += 1
        states[key] = [
            r6(result.values[s]),
            [r6(p) for p in result.row_policy[s]],
            [r6(p) for p in result.col_policy[s]],
            [[r6(M[i2, j]) for j in range(4)] for i2 in range(4)],
            [_outcome_repr(g, s, a0, a1, index) for a0, a1 in JOINT_ACTIONS],
            rounding,
        ]
    print(f"    {bid}: {rounding_flagged}/{len(states)} states have a "
          f"rounding-induced artifact at 3/2/1 decimals")

    return {
        "state_list": [f"{x0},{y0},{x1},{y1},{b}" for x0, y0, x1, y1, b in state_list],
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
