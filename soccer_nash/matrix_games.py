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
    """Best pure maximin row, breaking exact security ties by mean payoff.

    The secondary criterion prefers opportunities against opponent mistakes
    without sacrificing any worst-case value. A weakly dominated tied row
    cannot win this comparison. Remaining identical-score ties use action
    order; this selects one policy, not a unique equilibrium or a forced mix.
    """
    A = np.asarray(A, dtype=float)
    row_worst = A.min(axis=1)
    candidates = np.flatnonzero(row_worst == row_worst.max())
    i = int(candidates[np.argmax(A[candidates].mean(axis=1))])
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
        # Near-degenerate policy-iteration matrices can leave HiGHS presolve
        # with an unknown status. Retry the original constraints without
        # presolve; never accept an unsuccessful result or perturb payoffs.
        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
            method="highs-ipm", options={
                "presolve": False,
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
                "ipm_optimality_tolerance": 1e-10,
            },
        )
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


def _support_pairs(n: int):
    from itertools import combinations
    return [(rs, cs) for k in range(1, n + 1)
            for rs in combinations(range(n), k) for cs in combinations(range(n), k)]


def solve_zero_sum_batch(M: np.ndarray, tol: float = 1e-9):
    """Value and an equilibrium pair for every matrix in ``M`` (shape ``(N, n, n)``).

    Enumerates equal-size supports and solves the two indifference systems for
    all matrices at once, keeping the first support pair whose strategies are
    non-negative and unimprovable; the few matrices with no such pair (some
    degenerate games) fall back to the LP. Much faster than one LP per matrix
    for a network's mostly-mixed predictions."""
    M = np.asarray(M, dtype=float)
    N, n, _ = M.shape
    value = np.zeros(N)
    P, Q = np.zeros((N, n)), np.zeros((N, n))
    todo = np.ones(N, dtype=bool)
    for rs, cs in _support_pairs(n):
        idx = np.flatnonzero(todo)
        if len(idx) == 0:
            break
        k = len(rs)
        A = M[idx][:, rs][:, :, cs]
        rhs = np.zeros(k + 1)
        rhs[k] = 1.0
        sols, oks = [], []
        ones = np.broadcast_to(rhs[:, None], (len(idx), k + 1, 1))
        for B in (A.transpose(0, 2, 1), A):
            K = np.zeros((len(idx), k + 1, k + 1))
            K[:, :k, :k], K[:, :k, k], K[:, k, :k] = B, -1.0, 1.0
            ok = np.abs(np.linalg.det(K)) > 1e-9
            K[~ok] = np.eye(k + 1)
            sols.append(np.linalg.solve(K, ones)[..., 0])
            oks.append(ok)
        (x, v1), (y, v2) = (sols[0][:, :k], sols[0][:, k]), (sols[1][:, :k], sols[1][:, k])
        fx, fy = np.zeros((len(idx), n)), np.zeros((len(idx), n))
        fx[:, list(rs)], fy[:, list(cs)] = x, y
        Mi = M[idx]
        valid = (oks[0] & oks[1] & (x >= -tol).all(1) & (y >= -tol).all(1)
                 & (np.abs(v1 - v2) < 1e-7)
                 & (np.einsum("nij,nj->ni", Mi, fy).max(1) <= v1 + tol)
                 & (np.einsum("ni,nij->nj", fx, Mi).min(1) >= v1 - tol))
        hit = idx[valid]
        value[hit], P[hit], Q[hit] = v1[valid], fx[valid], fy[valid]
        todo[hit] = False
    for i in np.flatnonzero(todo):
        value[i], P[i], Q[i] = solve_zero_sum(M[i])
    return value, P, Q

