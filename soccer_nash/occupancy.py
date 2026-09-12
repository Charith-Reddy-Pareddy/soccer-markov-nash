"""State occupancy under a stationary joint policy.

A stochastic transition and a stochastic (mixed) policy both live in the
*occupancy distribution* -- the discounted share of time the
process spends in each state. This module computes it exactly (a linear fixpoint,
not sampling) and is used to ask whether the mixed-strategy states are actually
*on the equilibrium path* or just rare curiosities.

``visitation(game, row_policy, col_policy, gamma)`` returns

    d(s) = (1 - gamma) * sum_t gamma^t * P(S_t = s | S_0 = kickoff, pi)

normalised to sum to 1 over the non-terminal states the process can still be in
(mass that has already reached a goal in ``scoring="win"`` is dropped).
"""

from __future__ import annotations

import numpy as np

from soccer_nash.game import SoccerGame, State

Policy = dict[State, np.ndarray]


def _policy_transition_row(
    game: SoccerGame, s: State, row_policy: Policy, col_policy: Policy
) -> dict[State, float]:
    """P(next non-terminal state | s, pi) -- terminal mass is dropped."""
    p0 = np.asarray(row_policy[s], dtype=float)
    p1 = np.asarray(col_policy[s], dtype=float)
    acts = game.actions()
    out: dict[State, float] = {}
    for i, a0 in enumerate(acts):
        if p0[i] <= 0.0:
            continue
        for j, a1 in enumerate(acts):
            if p1[j] <= 0.0:
                continue
            w = p0[i] * p1[j]
            for prob, ns, _r in game.transitions(s, a0, a1):
                if game.is_terminal(ns):
                    continue
                out[ns] = out.get(ns, 0.0) + w * prob
    return out


def visitation(
    game: SoccerGame,
    row_policy: Policy,
    col_policy: Policy,
    gamma: float = 0.9,
    start: State | None = None,
    tol: float = 1e-12,
    max_iters: int = 100_000,
) -> dict[State, float]:
    """Discounted state-visitation distribution from ``start`` (default kickoff)."""
    s0 = start if start is not None else game.initial_state()
    states = list(game.states())
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)

    # sparse policy-induced transition, as row lists
    succ: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for s in states:
        row = _policy_transition_row(game, s, row_policy, col_policy)
        succ[idx[s]] = [(idx[ns], p) for ns, p in row.items()]

    mu0 = np.zeros(n)
    mu0[idx[s0]] = 1.0
    d = mu0.copy()
    flow = mu0.copy()
    for _ in range(max_iters):
        nxt = np.zeros(n)
        for i, mass in enumerate(flow):
            if mass == 0.0:
                continue
            for j, p in succ[i]:
                nxt[j] += mass * p
        flow = gamma * nxt
        d += flow
        if flow.sum() < tol:
            break

    d *= (1.0 - gamma)
    total = d.sum()
    if total > 0:
        d /= total
    return {states[i]: float(d[i]) for i in range(n) if d[i] > 0.0}


def concentration(
    dist: dict[State, float], subset: set[State]
) -> tuple[float, float]:
    """``(occupancy mass on subset, uniform share of subset)`` -- their ratio
    says how over- or under-represented the subset is on the equilibrium path."""
    mass = sum(dist.get(s, 0.0) for s in subset)
    uniform = len(subset) / len(dist) if dist else 0.0
    return mass, uniform
