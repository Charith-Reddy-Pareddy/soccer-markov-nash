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


def fp_step(M: np.ndarray, cr: np.ndarray, cc: np.ndarray) -> None:
    """One simultaneous fictitious-play round, updating the action counts
    ``cr`` / ``cc`` in place: each player best-responds to the other's
    empirical mix so far (ties -> lowest index)."""
    idx = np.arange(M.shape[0])
    q = cc / cc.sum(1, keepdims=True)
    p = cr / cr.sum(1, keepdims=True)
    i = np.argmax(np.einsum("nab,nb->na", M, q), axis=1)
    j = np.argmin(np.einsum("na,nab->nb", p, M), axis=1)
    cr[idx, i] += 1.0
    cc[idx, j] += 1.0


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
    for _ in range(1, rounds):
        fp_step(M, cr, cc)
    p = cr / cr.sum(1, keepdims=True)
    q = cc / cc.sum(1, keepdims=True)
    lo = np.einsum("na,nab->nb", p, M).min(1)
    hi = np.einsum("nab,nb->na", M, q).max(1)
    if single:
        return p[0], q[0], lo[0], hi[0]
    return p, q, lo, hi


def markov_fictitious_play(
    solver, exact_values: dict, sweeps: int, persistent: bool, rounds: int = 200,
    checkpoints: tuple[int, ...] = (),
):
    """Fictitious play inside the Markov game: every sweep rebuilds each state's
    4x4 stage matrix from the *current* value estimate ``V`` and updates ``V``
    from fictitious play on it -- no exact continuation value is used.

    ``persistent=True`` is the learning dynamic of the research note: the
    empirical action counts at every state **carry over across sweeps** and each
    sweep adds one best-response round (a player best-responds to the
    opponent's whole past play). ``persistent=False`` restarts the beliefs and
    runs ``rounds`` rounds each sweep. ``V`` is the midpoint of the value
    bounds ``lo <= val <= hi`` of the empirical mixes. Returns
    ``(V, history, (row_policy, col_policy))``: history rows are ``(sweep,
    max|V-V*|, mean bracket)`` and the policies are the final empirical mixes,
    ``{state: length-4 array}``.
    """
    game = solver.game
    states = list(game.states())
    V = dict.fromkeys(states, 0.0)
    v_star = np.array([exact_values[s] for s in states])
    n = solver._n
    cr = np.zeros((len(states), n))
    cc = np.zeros((len(states), n))
    cr[:, 0] = cc[:, 0] = 1.0
    history = []
    for k in range(1, sweeps + 1):
        M = np.array([solver._matrix(s, V) for s in states])
        if persistent:
            fp_step(M, cr, cc)
            p = cr / cr.sum(1, keepdims=True)
            q = cc / cc.sum(1, keepdims=True)
            lo = np.einsum("na,nab->nb", p, M).min(1)
            hi = np.einsum("nab,nb->na", M, q).max(1)
        else:
            p, q, lo, hi = fictitious_play(M, rounds)
        v = (lo + hi) / 2
        V = dict(zip(states, v.tolist()))
        if k in checkpoints or k == sweeps:
            history.append((k, float(np.abs(v - v_star).max()), float((hi - lo).mean())))
    policies = tuple(dict(zip(states, m)) for m in (p, q))
    return V, history, policies
