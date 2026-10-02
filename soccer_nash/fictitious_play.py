"""Best-response dynamics vs. fictitious play -- the research-group note's
"side quest" (rock-paper-scissors: best responses chase each other in a cycle,
the time-averaged play converges to the 1/3-1/3-1/3 equilibrium), then fictitious play
applied to the soccer game's stage games.

Fictitious play (Brown 1951; Robinson 1951 proved convergence for zero-sum
matrix games): each round, each player best-responds to the *empirical
average* of the opponent's past play. All routines are vectorised over a
leading batch axis, so every stage game of the soccer board runs at once.
"""

from __future__ import annotations

import numpy as np


def best_response_dynamics(M: np.ndarray, rounds: int, start=(0, 0)):
    """Both players play a pure best response to the opponent's *last* action.
    Returns the (row, col) action sequence."""
    i, j = start
    path = [(i, j)]
    for _ in range(rounds):
        ni = int(np.argmax(M[:, j]))
        nj = int(np.argmin(M[i, :]))
        i, j = ni, nj
        path.append((i, j))
    return path


def fictitious_play(M: np.ndarray, rounds: int):
    """Fictitious play on a batch of zero-sum matrix games ``M`` of shape
    ``(N, A, B)`` (row player maximises). Returns ``(p, q, lo, hi)``: the
    empirical mixed strategies and the guaranteed value bounds
    ``lo = min_j (pM)_j <= value <= hi = max_i (Mq)_i``."""
    M = np.asarray(M, dtype=float)
    single = M.ndim == 2
    if single:
        M = M[None]
    N, A, B = M.shape
    cr = np.zeros((N, A))
    cc = np.zeros((N, B))
    cr[:, 0] = 1.0
    cc[:, 0] = 1.0
    idx = np.arange(N)
    for t in range(1, rounds):
        q = cc / cc.sum(1, keepdims=True)
        p = cr / cr.sum(1, keepdims=True)
        i = np.argmax(np.einsum("nab,nb->na", M, q), axis=1)
        j = np.argmin(np.einsum("na,nab->nb", p, M), axis=1)
        cr[idx, i] += 1.0
        cc[idx, j] += 1.0
    p = cr / cr.sum(1, keepdims=True)
    q = cc / cc.sum(1, keepdims=True)
    lo = np.einsum("na,nab->nb", p, M).min(1)
    hi = np.einsum("nab,nb->na", M, q).max(1)
    if single:
        return p[0], q[0], lo[0], hi[0]
    return p, q, lo, hi
