"""Support enumeration for two-player normal-form games.

The zero-sum stage solver in :mod:`soccer_nash.matrix_games` uses a linear
program, which returns *one* equilibrium and needs a general-sum game to be
zero-sum. Support enumeration instead finds **every** Nash equilibrium of a
2-player game from its two payoff matrices, at the cost of trying every pair of
supports -- fine for the 4x4 stage games here.

``A`` is the row player's payoffs, ``B`` the column player's (``B = -A`` for a
zero-sum game). It also supplies the equilibrium-selection rule the research
notes mention -- pick the equilibrium with the largest sum of player values --
which for zero-sum games is a no-op because every equilibrium has value sum 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

_TOL = 1e-9


@dataclass(frozen=True)
class Equilibrium:
    row: np.ndarray  # row player's mixed strategy
    col: np.ndarray  # column player's mixed strategy
    row_value: float
    col_value: float

    @property
    def value_sum(self) -> float:
        return self.row_value + self.col_value


def _support_strategy(
    payoff_other: np.ndarray, own_support: list[int], other_support: list[int]
) -> np.ndarray | None:
    """Mix over ``own_support`` that makes the other player indifferent across
    ``other_support``. ``payoff_other[i, j]`` is the *other* player's payoff."""
    k = len(own_support)
    if k == 1:
        p = np.zeros(payoff_other.shape[0])
        p[own_support[0]] = 1.0
        return p

    sub = payoff_other[np.ix_(own_support, other_support)]  # k x k
    rows = [sub[:, r] - sub[:, r + 1] for r in range(k - 1)]
    rows.append(np.ones(k))
    C = np.array(rows)
    rhs = np.zeros(k)
    rhs[-1] = 1.0
    try:
        weights = np.linalg.solve(C, rhs)
    except np.linalg.LinAlgError:
        return None
    if np.any(weights < -_TOL):
        return None

    p = np.zeros(payoff_other.shape[0])
    p[own_support] = np.clip(weights, 0.0, None)
    s = p.sum()
    return p / s if s > _TOL else None


def all_equilibria(
    A: np.ndarray, B: np.ndarray, tol: float = 1e-7
) -> list[Equilibrium]:
    """Every Nash equilibrium found by support enumeration."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, n = A.shape

    found: list[Equilibrium] = []
    for size in range(1, min(m, n) + 1):
        for rows in combinations(range(m), size):
            for cols in combinations(range(n), size):
                rows_l, cols_l = list(rows), list(cols)
                p = _support_strategy(B, rows_l, cols_l)
                q = _support_strategy(A.T, cols_l, rows_l)
                if p is None or q is None:
                    continue

                row_payoffs = A @ q
                col_payoffs = p @ B
                row_val = float(row_payoffs[rows_l[0]])
                col_val = float(col_payoffs[cols_l[0]])

                # No profitable deviation for either player.
                if row_payoffs.max() > row_val + tol:
                    continue
                if col_payoffs.max() > col_val + tol:
                    continue
                # Support entries really are optimal (not just the first one).
                if np.any(np.abs(row_payoffs[rows_l] - row_val) > tol):
                    continue
                if np.any(np.abs(col_payoffs[cols_l] - col_val) > tol):
                    continue

                if not any(
                    np.allclose(p, e.row, atol=1e-6)
                    and np.allclose(q, e.col, atol=1e-6)
                    for e in found
                ):
                    found.append(Equilibrium(p, q, row_val, col_val))
    return found


def select_equilibrium(
    equilibria: list[Equilibrium], rule: str = "largest_sum"
) -> Equilibrium:
    """Pick one equilibrium. ``"largest_sum"`` maximises ``row_value +
    col_value`` (a no-op for zero-sum); ``"row"`` favours the row player."""
    if not equilibria:
        raise ValueError("no equilibria")
    if rule == "largest_sum":
        return max(equilibria, key=lambda e: e.value_sum)
    if rule == "row":
        return max(equilibria, key=lambda e: e.row_value)
    raise ValueError(f"unknown rule {rule!r}")


def zero_sum_value(A: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """``(value, row_strategy, col_strategy)`` via support enumeration, for
    cross-checking the LP solver."""
    A = np.asarray(A, dtype=float)
    eqs = all_equilibria(A, -A)
    if not eqs:
        raise RuntimeError("support enumeration found no equilibrium")
    e = eqs[0]
    return e.row_value, e.row, e.col
