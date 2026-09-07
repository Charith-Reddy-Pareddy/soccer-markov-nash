"""Stage-game solvers for two-player zero-sum games.

Convention: ``A`` is the row player's payoff matrix. The row player (player 0)
maximises; the column player (player 1) minimises. Player 1's payoff is ``-A``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

_TOL = 1e-9


def pure_bounds(A: np.ndarray) -> tuple[float, float]:
    """Return ``(maximin, minimax)`` over pure strategies.

    A pure-strategy saddle point exists iff the two are equal.
    """
    A = np.asarray(A, dtype=float)
    maximin = A.min(axis=1).max()
    minimax = A.max(axis=0).min()
    return float(maximin), float(minimax)


def has_pure_saddle(A: np.ndarray) -> bool:
    lo, hi = pure_bounds(A)
    return abs(hi - lo) <= _TOL


def pure_saddle_points(A: np.ndarray) -> list[tuple[int, int]]:
    """All pure-strategy saddle points ``(i, j)`` of the zero-sum game."""
    A = np.asarray(A, dtype=float)
    col_max = A.max(axis=0)
    row_min = A.min(axis=1)
    out: list[tuple[int, int]] = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] >= col_max[j] - _TOL and A[i, j] <= row_min[i] + _TOL:
                out.append((i, j))
    return out


def security_strategy_row(A: np.ndarray) -> tuple[int, float]:
    """Best pure maximin row and its guaranteed value for the row player."""
    A = np.asarray(A, dtype=float)
    row_worst = A.min(axis=1)
    i = int(np.argmax(row_worst))
    return i, float(row_worst[i])


def _lp_row_value(A: np.ndarray) -> tuple[float, np.ndarray]:
    """Maximin mixed strategy for the row player of ``A`` via linear program."""
    A = np.asarray(A, dtype=float)
    n, m = A.shape

    # Variables: p_0..p_{n-1}, v.  Minimise -v.
    c = np.zeros(n + 1)
    c[-1] = -1.0

    # For every column j:  v - sum_i p_i A[i, j] <= 0
    A_ub = np.hstack([-A.T, np.ones((m, 1))])
    b_ub = np.zeros(m)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)] * n + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    p = np.clip(res.x[:n], 0.0, None)
    p = p / p.sum()
    return float(res.x[-1]), p


def game_value(A: np.ndarray) -> float:
    """Row player's minimax value only (one LP, no strategies)."""
    return _lp_row_value(np.asarray(A, dtype=float))[0]


def solve_zero_sum(A: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``(value, row_strategy, col_strategy)`` for the zero-sum game.

    ``value`` is the row player's game value; ``col_strategy`` is player 1's
    minimising mixed strategy.
    """
    A = np.asarray(A, dtype=float)
    value, p = _lp_row_value(A)
    # Player 1 maximises -A^T as a row player.
    _, q = _lp_row_value(-A.T)
    return value, p, q
