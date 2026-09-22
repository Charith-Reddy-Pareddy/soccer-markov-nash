"""Numerical foundations for mixed Nash Q-iteration on zero-sum stage games.

Three problems come up once a stage game genuinely needs mixed strategies:

1. **The value is three numbers.** For candidate strategies ``(p, q)`` the row
   player can *guarantee* ``lower = min_j (pM)_j``, can *reach at most*
   ``upper = max_i (Mq)_i`` if it best-responds, and *gets* ``mid = p M q`` if
   both mix. At an exact equilibrium these coincide; numerically they never do.
   :func:`value_bracket` returns all three. The true minimax value always lies
   in ``[lower, upper]`` (weak duality), so ``gap = upper - lower`` is a
   *certified* bound on the error of the LP solution.

2. **Rounding to force indifference is scale-blind.** A stage game that is
   essentially rock-paper-scissors plus a tiny epsilon has no exact pure
   saddle, but rounding its entries to (say) one decimal creates a spurious
   one -- and with discounting the real ``gamma^k`` value differences are
   themselves smaller than that. :func:`classify_stage_game` uses a tolerance
   that scales with the entries instead, and :func:`rounding_changes_saddle`
   shows where fixed rounding would flip the answer.

3. **Where does the game "have to" mix, and how much?** :func:`essential_subgame`
   strips iteratively strictly dominated actions; :func:`is_matching_pennies`
   checks whether what remains is a no-pure-saddle square game;
   :func:`mixing_entropy` measures *how* mixed a strategy is (0 = pure, 1 bit =
   an even 2-way mix); :func:`epsilon_equilibrium` measures how far a strategy
   pair is from a Nash.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from soccer_nash.matrix_games import pure_bounds, solve_zero_sum

_ABS = 1e-9


class ValueBracket(NamedTuple):
    lower: float  # row player's guaranteed value if it commits to p
    mid: float  # p^T M q
    upper: float  # most the row player can get if it best-responds to q

    @property
    def gap(self) -> float:
        return self.upper - self.lower


def value_bracket(
    M: np.ndarray, p: np.ndarray, q: np.ndarray
) -> ValueBracket:
    """The three quantities for the row (maximising) player."""
    M = np.asarray(M, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return ValueBracket(
        lower=float((p @ M).min()),
        # M[i, j] is the row player's payoff, so the mixed value is p^T M q.
        # (The notes' pi_2^T Q pi_1 uses the transposed layout.)
        mid=float(p @ M @ q),
        upper=float((M @ q).max()),
    )


def epsilon_equilibrium(M: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """How much either player can gain by deviating from ``(p, q)``.

    ``r_row = max_i (M q)_i - p^T M q`` (row player's best deviation),
    ``r_col = p^T M q - min_j (p^T M)_j`` (column player's), and
    ``eps = max(r_row, r_col)``. ``0`` exactly at a Nash equilibrium; a clean
    equilibrium-quality number for the LP, the pure-first hybrid, and the neural
    approximation alike -- more general than exploitability, which needs a full
    best-response solve.
    """
    b = value_bracket(M, p, q)
    return max(b.upper - b.mid, b.mid - b.lower)


def mixing_entropy(p: np.ndarray, base: float = 2.0, tol: float = 1e-9) -> float:
    """Shannon entropy of a strategy, in bits by default. ``0`` for a pure
    strategy; ``1`` for an even 2-way mix (matching pennies); higher for a
    genuinely spread mixed strategy. Distinguishes a near-pure hedge from real
    randomisation."""
    p = np.asarray(p, dtype=float)
    p = p[p > tol]
    if len(p) <= 1:
        return 0.0
    p = p / p.sum()
    return float(-(p * (np.log(p) / np.log(base))).sum())


def certified_value(M: np.ndarray) -> tuple[float, float]:
    """``(value, error)`` with a guarantee that ``|value - minimax(M)| <= error``."""
    M = np.asarray(M, dtype=float)
    _, p, q = solve_zero_sum(M)
    b = value_bracket(M, p, q)
    return b.mid, b.gap


def classify_stage_game(M: np.ndarray, rel_tol: float = 1e-6) -> str:
    """One of ``"pure"``, ``"mixed"``, ``"degenerate"``.

    The full taxonomy has five cases; this function collapses the first three
    ("does a pure saddle exist, and is it sharp") into ``"pure"`` /
    ``"degenerate"`` and reports the last as ``"mixed"``:

    1. *exact pure saddle* -- ``maximin == minimax`` with no tolerance;
    2. *numerically-indistinguishable saddle* -- ``minimax - maximin`` is below
       the scaled tolerance ``rel_tol * max(1, |M|max)``. With discounting the
       real ``gamma^k`` gaps can themselves be tiny, so a fixed absolute
       tolerance would wrongly merge cases 2 and 5 (see
       :func:`rounding_changes_saddle`);
    3. *strict pure saddle* -> returns ``"pure"`` -- cases 1/2 **and** a unique
       row attains the maximin and a unique column the minimax;
    4. *degenerate saddle* -> returns ``"degenerate"`` -- cases 1/2 but several
       rows or columns tie (e.g. the all-zeros game). Still an equilibrium to
       play the saddle, but rounding could turn it into matching pennies;
    5. *genuine mixed* -> returns ``"mixed"`` -- ``minimax - maximin`` exceeds
       the scaled tolerance; an LP is genuinely required.

    Only case 5 forces the pure-first hybrid onto its LP branch.
    """
    M = np.asarray(M, dtype=float)
    maximin, minimax = pure_bounds(M)
    scale = max(1.0, float(np.abs(M).max()))
    tol = rel_tol * scale
    if minimax - maximin > tol:
        return "mixed"

    value = 0.5 * (maximin + minimax)
    near_maximin = int((M.min(axis=1) >= value - tol).sum())
    near_minimax = int((M.max(axis=0) <= value + tol).sum())
    if near_maximin == 1 and near_minimax == 1:
        return "pure"
    return "degenerate"


def rounding_changes_saddle(M: np.ndarray, decimals: int = 1) -> bool:
    """Does rounding to ``decimals`` places flip whether a pure saddle exists?"""
    M = np.asarray(M, dtype=float)
    exact = (lambda lo, hi: hi - lo > _ABS)(*pure_bounds(M))
    rounded = (lambda lo, hi: hi - lo > _ABS)(*pure_bounds(np.round(M, decimals)))
    return exact != rounded


class RoundingDiagnostic(NamedTuple):
    decimals: int | None  # None = full, unrounded precision
    pure: bool
    value: float
    row: np.ndarray
    col: np.ndarray


def rounding_diagnostic(
    M: np.ndarray, decimals: tuple[int | None, ...] = (None, 3, 2, 1)
) -> list[RoundingDiagnostic]:
    """Solve ``M`` fresh at each precision in ``decimals`` (``None`` = full,
    unrounded precision) and report both whether the *rounded* game has a
    pure saddle and what its own equilibrium actually is -- not just
    whether classification flips (:func:`rounding_changes_saddle`'s
    question), the policy itself, since that is the concrete thing "does
    rounding the displayed matrix change the reported mix" asks.

    This makes literal the "we definitely need some kind of approximation
    rounding" request (the project's Sept research meetings): round the
    *displayed* matrix, then solve that rounded matrix as its own game and
    see what comes out, rather than reasoning about the exact value with a
    scale-blind tolerance (that is what :func:`classify_stage_game` already
    avoids doing). There is no single "correct" precision -- on
    ``docs/positions.md``'s Case 3, rounding to 2 decimals reproduces the
    exact mix to within a percentage point, and rounding to 1 decimal
    manufactures a pure saddle that is not actually there. Both are
    legitimate outputs of *this* function; picking one as "the" rounding
    convention is a modeling choice this function deliberately does not
    make for the caller.
    """
    M = np.asarray(M, dtype=float)
    out = []
    for d in decimals:
        Mr = M if d is None else np.round(M, d)
        pure = not (lambda lo, hi: hi - lo > _ABS)(*pure_bounds(Mr))
        value, row, col = solve_zero_sum(Mr)
        out.append(RoundingDiagnostic(decimals=d, pure=pure, value=value, row=row, col=col))
    return out


def _support_set(policy: np.ndarray, tol: float = 1e-6) -> frozenset[int]:
    return frozenset(int(i) for i, p in enumerate(policy) if p > tol)


def is_rounding_artifact(
    full: RoundingDiagnostic, rounded: RoundingDiagnostic, tol: float = 1e-6
) -> bool:
    """Is ``rounded`` a genuine rounding artifact relative to ``full``?

    True when rounding changes the pure/mixed classification, or changes
    *which* actions carry non-trivial probability for either player --
    never for a percentage-split drift within an unchanged support (e.g.
    63.5% -> 66.7% on the same action is not an artifact; that action
    dropping out of the support is).
    """
    if full.pure != rounded.pure:
        return True
    return (
        _support_set(full.row, tol) != _support_set(rounded.row, tol)
        or _support_set(full.col, tol) != _support_set(rounded.col, tol)
    )


def essential_subgame(
    M: np.ndarray, tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove iteratively strictly dominated rows/columns.

    Returns ``(row_indices, col_indices, reduced_matrix)``. Rows are the
    maximiser's actions, columns the minimiser's. Value- and equilibrium-
    preserving, but only strict dominance -- games with tied actions barely
    shrink.
    """
    M = np.asarray(M, dtype=float)
    rows = list(range(M.shape[0]))
    cols = list(range(M.shape[1]))

    changed = True
    while changed and len(rows) > 1 and len(cols) > 1:
        changed = False
        sub = M[np.ix_(rows, cols)]

        for a in range(len(rows)):
            if any(
                b != a and np.all(sub[b] > sub[a] + tol) for b in range(len(rows))
            ):
                del rows[a]
                changed = True
                break
        if changed:
            continue

        sub = M[np.ix_(rows, cols)]
        for a in range(len(cols)):
            if any(
                b != a and np.all(sub[:, b] < sub[:, a] - tol)
                for b in range(len(cols))
            ):
                del cols[a]
                changed = True
                break

    idx_r = np.array(rows)
    idx_c = np.array(cols)
    return idx_r, idx_c, M[np.ix_(idx_r, idx_c)]


def support_shape(M: np.ndarray, tol: float = 1e-6) -> tuple[int, int]:
    """``(row_support, col_support)`` of the LP equilibrium of ``M``.

    ``(2, 2)`` means the equilibrium is a matching-pennies mix over two
    actions each -- the smallest genuinely mixed structure.
    """
    _, p, q = solve_zero_sum(np.asarray(M, dtype=float))
    return int((p > tol).sum()), int((q > tol).sum())


def is_matching_pennies(M: np.ndarray, tol: float = 1e-9) -> bool:
    """True if ``M`` is square with no pure saddle -- a generalised
    matching-pennies / rock-paper-scissors structure."""
    M = np.asarray(M, dtype=float)
    if M.shape[0] != M.shape[1] or M.shape[0] < 2:
        return False
    maximin, minimax = pure_bounds(M)
    return minimax - maximin > tol
