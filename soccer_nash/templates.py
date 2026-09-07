"""Reduce the mixed-strategy states to a handful of geometric templates.

The pipeline is::

    no-pure-saddle states
        -> mirror/role-swap canonicalization         (halves the set)
        -> cluster by carrier-frame geometry          (a few templates)
        -> per template: the equilibrium-support subgame, its
           matching-pennies pattern, and the {U, D, L, R}
           meaning of each support action

so "the mixed states are matching pennies" can be shown rather than asserted,
and the report can print one stage matrix per mechanism instead of 94.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soccer_nash.game import SoccerGame, State
from soccer_nash.geometry import features
from soccer_nash.matrix_games import solve_zero_sum
from soccer_nash.symmetry import mirror_state

_ACTION_NAME = {0: "U", 1: "D", 2: "L", 3: "R"}
_DELTA = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}


# --------------------------------------------------------------------- symmetry


def mirror_reduce(game: SoccerGame, states: list[State]) -> list[State]:
    """One representative per mirror/role-swap pair, order preserved.

    The mirror map flips the board left-right and swaps the players, so a state
    and its image have identical carrier-frame geometry.
    """
    seen: set[State] = set()
    reps: list[State] = []
    for s in states:
        if s in seen:
            continue
        seen.add(s)
        seen.add(mirror_state(s, game.width))
        reps.append(s)
    return reps


# ----------------------------------------------------------- action semantics


def describe_action(game: SoccerGame, state: State, mover: int, action: int) -> str:
    """A geometry-derived gloss for one player's action in ``state``.

    Carrier actions are named by their effect on the attack; defender actions by
    where they land relative to the carrier and the carrier's forward
    "threat" cell ``(cx + forward, cy)``.
    """
    x0, y0, x1, y1, b = state
    (cx, cy), (dx, dy) = ((x0, y0), (x1, y1)) if b == 0 else ((x1, y1), (x0, y0))
    forward = 1 if b == 0 else -1
    threat = (cx + forward, cy)
    ddx, ddy = _DELTA[action]

    if mover == b:  # the carrier
        if ddx == forward:
            return "advance"
        if ddx == -forward:
            return "retreat"
        return "climb" if ddy > 0 else "drop"

    tgt = (dx + ddx, dy + ddy)  # the defender
    if tgt == threat:
        return "block"
    if tgt == (cx, cy):
        return "tackle"
    if ddx == -forward:
        return "step in"
    if ddx == forward:
        return "drop back"
    return "cover up" if ddy > 0 else "cover down"


# ------------------------------------------------------ matching-pennies check


@dataclass(frozen=True)
class MatchingPennies:
    """A 2x2 zero-sum game and its (row-max / col-min) best-response structure."""

    a: float
    b: float
    c: float
    d: float
    row_crosses: bool  # row's best reply to col 0 differs from its reply to col 1
    col_crosses: bool  # col's best reply to row 0 differs from its reply to row 1
    p_row: float  # equilibrium weight on row 0
    q_col: float  # equilibrium weight on col 0
    value: float

    @property
    def is_matching_pennies(self) -> bool:
        return self.row_crosses and self.col_crosses


def matching_pennies_pattern(M2: np.ndarray, tol: float = 1e-9) -> MatchingPennies:
    """Classify a 2x2 ``[[a, b], [c, d]]`` (row maximises, column minimises)."""
    a, b = float(M2[0, 0]), float(M2[0, 1])
    c, d = float(M2[1, 0]), float(M2[1, 1])

    # Best replies "cross": the argmax row flips between the two columns, and
    # the argmin column flips between the two rows -- exactly no pure saddle.
    row_crosses = (a - c > tol) != (b - d > tol)
    col_crosses = (a - b < -tol) != (c - d < -tol)

    denom = a - b - c + d
    if abs(denom) < 1e-12:
        p_row = q_col = value = float("nan")
    else:
        p_row = (d - c) / denom
        q_col = (d - b) / denom
        value = (a * d - b * c) / denom
    return MatchingPennies(a, b, c, d, row_crosses, col_crosses, p_row, q_col, value)


# ------------------------------------------------------------------- templates


def equilibrium_support(
    M: np.ndarray, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(row_idx, col_idx, submatrix)`` -- the actions each player plays with
    positive probability at the LP equilibrium of the zero-sum game ``M``."""
    M = np.asarray(M, dtype=float)
    _, p, q = solve_zero_sum(M)
    ri = np.flatnonzero(p > tol)
    ci = np.flatnonzero(q > tol)
    return ri, ci, M[np.ix_(ri, ci)]


@dataclass
class Template:
    key: tuple  # (support shape, defender dx_rel, |dy_rel|, carrier on goal row)
    count: int  # mixed states in this class (before mirror reduction)
    representative: State
    subgame: np.ndarray  # equilibrium-support matrix, carrier = maximising rows
    carrier_actions: list[tuple[str, str]]  # (U/D/L/R, gloss)
    defender_actions: list[tuple[str, str]]
    mp: MatchingPennies | None  # set when the support is 2x2


def _carrier_frame(
    state: State, ri: np.ndarray, ci: np.ndarray, sub: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorient a player-0 payoff submatrix so the carrier is the maximising
    row player, whichever player holds the ball."""
    if state[4] == 0:
        return ri, ci, sub
    return ci, ri, -sub.T


def _geometry_key(f: dict, carrier_support: tuple[int, int]) -> tuple:
    dx = f["dx_rel"] if f["dx_rel"] <= 2 else 3          # defender's forward lead
    dy = min(f["abs_dy_rel"], 2)
    return (carrier_support, dx, dy, f["carrier_in_goal_row"])


def mixed_state_templates(
    game: SoccerGame,
    no_saddle_states: list[State],
    matrix_of,
) -> list[Template]:
    """Group ``no_saddle_states`` into geometric templates.

    ``matrix_of(state) -> 4x4 payoff matrix`` is ``NashQIteration._matrix``
    bound to the converged values (passed in so this module stays solver-free).
    """
    buckets: dict[tuple, list[State]] = {}
    frames: dict[State, tuple] = {}
    for s in no_saddle_states:
        ri, ci, sub = equilibrium_support(matrix_of(s))
        cr, cd, csub = _carrier_frame(s, ri, ci, sub)
        frames[s] = (cr, cd, csub)
        buckets.setdefault(
            _geometry_key(features(game, s), csub.shape), []
        ).append(s)

    mid_x, mid_y = (game.width - 1) / 2, (game.height - 1) / 2

    def centrality(s: State) -> float:
        x0, y0, x1, y1, b = s
        cx, cy = (x0, y0) if b == 0 else (x1, y1)
        return abs(cx - mid_x) + abs(cy - mid_y)

    templates: list[Template] = []
    for key, states in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        # the most central state in the class is the clearest exemplar
        rep = min(mirror_reduce(game, states), key=centrality)
        carrier, defender = rep[4], 1 - rep[4]
        cr, cd, csub = frames[rep]
        carrier_actions = [
            (_ACTION_NAME[int(i)], describe_action(game, rep, carrier, int(i)))
            for i in cr
        ]
        defender_actions = [
            (_ACTION_NAME[int(j)], describe_action(game, rep, defender, int(j)))
            for j in cd
        ]
        mp = matching_pennies_pattern(csub) if csub.shape == (2, 2) else None
        templates.append(
            Template(key, len(states), rep, csub, carrier_actions, defender_actions, mp)
        )
    return templates


def format_templates(templates: list[Template]) -> str:
    """A plain-text report of the templates, for scripts and docs."""
    lines: list[str] = []
    total = sum(t.count for t in templates)
    lines.append(f"{len(templates)} templates cover {total} mixed states\n")
    for n, t in enumerate(templates, 1):
        shape, dx, dy, in_row = t.key
        lines.append(
            f"[{n}] {t.count:>3d} states  --  support {shape[0]}x{shape[1]}, "
            f"defender {dx} ahead, |dy|={dy}, "
            f"carrier {'on' if in_row else 'off'} a goal row"
        )
        lines.append(f"     representative {t.representative}")
        lines.append(
            "     carrier:  " + ", ".join(f"{a}={g}" for a, g in t.carrier_actions)
        )
        lines.append(
            "     defender: " + ", ".join(f"{a}={g}" for a, g in t.defender_actions)
        )
        with np.printoptions(precision=3, suppress=True):
            for row in np.atleast_2d(t.subgame):
                lines.append(f"       {np.array2string(row)}")
        if t.mp is not None:
            tag = (
                "matching pennies"
                if t.mp.is_matching_pennies
                else "2x2, best replies do not cross"
            )
            lines.append(
                f"     {tag}: p*={t.mp.p_row:.2f} on row 0, "
                f"q*={t.mp.q_col:.2f} on col 0, value {t.mp.value:.3f}"
            )
        lines.append("")
    return "\n".join(lines)
