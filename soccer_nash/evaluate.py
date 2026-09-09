"""Exact expected return of a fixed pair of stationary policies.

Both players' moves are pinned, so the game is a Markov reward process and its
value is a linear fixpoint

    V(s) = sum_{a0,a1} row(s)[a0] col(s)[a1] * sum_s' P(s'|s,a0,a1) (r0 + gamma V(s'))

solved by iteration. This gives the row player's expected discounted return with
no sampling noise -- used to score policies against fixed opponents in the
tournament (`scripts/tournament.py`).
"""

from __future__ import annotations

import numpy as np

from soccer_nash.game import SoccerGame, State

Policy = dict[State, np.ndarray]


def policy_value(
    game: SoccerGame,
    row_policy: Policy,
    col_policy: Policy,
    gamma: float = 0.9,
    tol: float = 1e-12,
    max_iters: int = 100_000,
) -> dict[State, float]:
    """Row player's expected discounted return from every state under the
    joint stationary policy ``(row_policy, col_policy)``."""
    states = list(game.states())
    acts = game.actions()

    # Precompute, per state, the list of (weight, next_state_or_None, reward).
    backups: dict[State, list[tuple[float, State | None, float]]] = {}
    for s in states:
        p0 = np.asarray(row_policy[s], dtype=float)
        p1 = np.asarray(col_policy[s], dtype=float)
        entries: list[tuple[float, State | None, float]] = []
        for i, a0 in enumerate(acts):
            if p0[i] <= 0.0:
                continue
            for j, a1 in enumerate(acts):
                if p1[j] <= 0.0:
                    continue
                w = p0[i] * p1[j]
                for prob, ns, (r0, _r1) in game.transitions(s, a0, a1):
                    if game.is_terminal(ns):
                        entries.append((w * prob, None, r0))
                    else:
                        entries.append((w * prob, ns, r0))
        backups[s] = entries

    values = dict.fromkeys(states, 0.0)
    for _ in range(max_iters):
        delta = 0.0
        updated: dict[State, float] = {}
        for s in states:
            v = 0.0
            for w, ns, r in backups[s]:
                v += w * (r if ns is None else r + gamma * values[ns])
            updated[s] = v
            delta = max(delta, abs(v - values[s]))
        values = updated
        if delta < tol:
            break
    return values
