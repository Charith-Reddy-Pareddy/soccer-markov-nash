"""General-sum two-player Markov game solver via Nash value iteration.

The zero-sum solver in :mod:`soccer_nash.nash_q` uses an LP and takes the
minimax value. This one supports arbitrary (non-zero-sum) rewards: at each
state it enumerates every stage-game equilibrium with
:func:`soccer_nash.support_enum.all_equilibria` and selects one with a rule
(the research notes' "largest sum of values"). It is deliberately small and
generic -- the game is given as plain callables.

    transition(s, a0, a1) -> list of (probability, next_state)
    reward(s, a0, a1, s')  -> (r0, r1)

Terminal states are any not returned by ``states()``; their value is 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Sequence

import numpy as np

from soccer_nash.support_enum import all_equilibria, select_equilibrium

State = Hashable
Transition = Callable[[State, int, int], Sequence[tuple[float, State]]]
Reward = Callable[[State, int, int, State], tuple[float, float]]


@dataclass
class MarkovGameResult:
    row_values: dict[State, float]
    col_values: dict[State, float]
    row_policy: dict[State, np.ndarray]
    col_policy: dict[State, np.ndarray]
    iterations: int
    multi_equilibrium_states: list[State]


def solve_markov_game(
    states: Sequence[State],
    n_actions: tuple[int, int],
    transition: Transition,
    reward: Reward,
    gamma: float = 0.9,
    select: str = "largest_sum",
    tol: float = 1e-8,
    max_iters: int = 2000,
) -> MarkovGameResult:
    states = list(states)
    known = set(states)
    n0, n1 = n_actions

    v0 = {s: 0.0 for s in states}
    v1 = {s: 0.0 for s in states}

    iterations = 0
    for iterations in range(1, max_iters + 1):
        delta = 0.0
        nv0: dict[State, float] = {}
        nv1: dict[State, float] = {}
        for s in states:
            A = np.zeros((n0, n1))
            B = np.zeros((n0, n1))
            for a0 in range(n0):
                for a1 in range(n1):
                    for prob, s_next in transition(s, a0, a1):
                        r0, r1 = reward(s, a0, a1, s_next)
                        cont0 = v0[s_next] if s_next in known else 0.0
                        cont1 = v1[s_next] if s_next in known else 0.0
                        A[a0, a1] += prob * (r0 + gamma * cont0)
                        B[a0, a1] += prob * (r1 + gamma * cont1)
            eq = select_equilibrium(all_equilibria(A, B), rule=select)
            nv0[s] = eq.row_value
            nv1[s] = eq.col_value
            delta = max(delta, abs(nv0[s] - v0[s]), abs(nv1[s] - v1[s]))
        v0, v1 = nv0, nv1
        if delta < tol:
            break

    row_policy: dict[State, np.ndarray] = {}
    col_policy: dict[State, np.ndarray] = {}
    multi: list[State] = []
    for s in states:
        A = np.zeros((n0, n1))
        B = np.zeros((n0, n1))
        for a0 in range(n0):
            for a1 in range(n1):
                for prob, s_next in transition(s, a0, a1):
                    r0, r1 = reward(s, a0, a1, s_next)
                    cont0 = v0[s_next] if s_next in known else 0.0
                    cont1 = v1[s_next] if s_next in known else 0.0
                    A[a0, a1] += prob * (r0 + gamma * cont0)
                    B[a0, a1] += prob * (r1 + gamma * cont1)
        eqs = all_equilibria(A, B)
        if len(eqs) > 1:
            multi.append(s)
        eq = select_equilibrium(eqs, rule=select)
        row_policy[s] = eq.row
        col_policy[s] = eq.col

    return MarkovGameResult(v0, v1, row_policy, col_policy, iterations, multi)
