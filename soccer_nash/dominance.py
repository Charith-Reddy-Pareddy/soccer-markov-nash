"""Iterated elimination of weakly dominated actions for a zero-sum stage game.

``essential_subgame`` in :mod:`soccer_nash.numerics` removes *strictly*
dominated actions (value-preserving, but it barely shrinks the soccer stage
games). This module removes *weakly* dominated ones, which is exactly the
reduction that certifies a pure saddle: a stage game is solvable this way iff
its residual has a pure-strategy saddle point, and for the single-goal-cell
soccer game every converged stage game is.
"""

from __future__ import annotations

import numpy as np

from soccer_nash.matrix_games import pure_bounds

_TOL = 1e-7


def iewds(M: np.ndarray, tol: float = _TOL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively drop weakly dominated rows/cols.

    Row player maximises, column player minimises. Returns
    ``(row_idx, col_idx, residual)``. The order of elimination can matter for
    weak dominance; this uses a fixed rule (rows before columns, lowest index
    first) so the result is reproducible.
    """
    M = np.asarray(M, dtype=float)
    rows = list(range(M.shape[0]))
    cols = list(range(M.shape[1]))

    changed = True
    while changed and (len(rows) > 1 or len(cols) > 1):
        changed = False
        A = M[np.ix_(rows, cols)]
        for a in range(len(rows)):
            if any(b != a and np.all(A[b] >= A[a] - tol) for b in range(len(rows))):
                del rows[a]
                changed = True
                break
        if changed:
            continue
        A = M[np.ix_(rows, cols)]
        for a in range(len(cols)):
            if any(
                b != a and np.all(A[:, b] <= A[:, a] + tol)
                for b in range(len(cols))
            ):
                del cols[a]
                changed = True
                break

    ri = np.array(rows)
    ci = np.array(cols)
    return ri, ci, M[np.ix_(ri, ci)]


def is_iewds_solvable(M: np.ndarray, tol: float = _TOL) -> bool:
    """True if iterated weak-dominance elimination leaves a game with a pure
    saddle point (a 1xk / kx1 / 1x1 residual counts)."""
    _, _, residual = iewds(M, tol)
    lo, hi = pure_bounds(residual)
    return hi - lo <= tol
