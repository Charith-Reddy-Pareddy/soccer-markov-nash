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

3. **Where does the game "have to" mix?** :func:`essential_subgame` strips
   iteratively strictly dominated actions; :func:`is_matching_pennies` checks
   whether what remains is a no-pure-saddle square game.
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


def certified_value(M: np.ndarray) -> tuple[float, float]:
    """``(value, error)`` with a guarantee that ``|value - minimax(M)| <= error``."""
    M = np.asarray(M, dtype=float)
    _, p, q = solve_zero_sum(M)
    b = value_bracket(M, p, q)
    return b.mid, b.gap


def classify_stage_game(M: np.ndarray, rel_tol: float = 1e-6) -> str:
    """One of ``"pure"``, ``"mixed"``, ``"degenerate"``.

    * ``mixed``      -- ``minimax - maximin`` exceeds the scaled tolerance: no
      pure saddle, mixing is genuinely required.
    * ``pure``       -- a pure saddle exists and it is *strict*: a unique row
      attains the maximin and a unique column the minimax, each with margin.
    * ``degenerate`` -- a pure saddle exists but is not strict (several near-tied
      rows or columns, e.g. the all-zeros game). Playing the saddle is still an
      equilibrium, but rounding could turn it into a matching-pennies game.

    The tolerance scales with the matrix entries, so it behaves the same on a
    stage game deep in the discounted ``gamma^k`` bands as on one near the goal.
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
