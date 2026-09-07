"""General-sum two-player Markov game solver via Nash value iteration.

The zero-sum solver in :mod:`soccer_nash.nash_q` uses an LP and takes the
minimax value. This one supports arbitrary (non-zero-sum) rewards. It keeps the
project's *pure-first* philosophy: at each state it looks for a pure Nash
equilibrium (one cell that is a mutual best response, `O(A^2)`) and only calls
support enumeration (:func:`soccer_nash.support_enum.all_equilibria`) when there
is none. Among several equilibria it applies a selection rule -- the research
notes' "largest sum of values".

The game is given as plain callables::

    transition(s, a0, a1) -> list of (probability, next_state)
    reward(s, a0, a1, s')  -> (r0, r1)

Terminal states are any not in ``states``; their value is 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Sequence

import numpy as np

from soccer_nash.support_enum import Equilibrium, all_equilibria, select_equilibrium

State = Hashable
Transition = Callable[[State, int, int], Sequence[tuple[float, State]]]
Reward = Callable[[State, int, int, State], tuple[float, float]]

_TOL = 1e-9


@dataclass
class MarkovGameResult:
    row_values: dict[State, float]
    col_values: dict[State, float]
    row_policy: dict[State, np.ndarray]
    col_policy: dict[State, np.ndarray]
    iterations: int
    multi_equilibrium_states: list[State]
    support_enum_states: list[State]


def _pure_nash(A: np.ndarray, B: np.ndarray) -> list[tuple[int, int]]:
    """Cells that are a mutual best response."""
    col_best = A.max(axis=0)  # row player's best payoff per column
    row_best = B.max(axis=1)  # column player's best payoff per row
    out = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] >= col_best[j] - _TOL and B[i, j] >= row_best[i] - _TOL:
                out.append((i, j))
    return out


def _equilibria(A: np.ndarray, B: np.ndarray) -> tuple[list[Equilibrium], bool]:
    """Equilibria of the stage game, plus whether support enumeration was used."""
    pures = _pure_nash(A, B)
    if pures:
        n0, n1 = A.shape
        eqs = []
        for i, j in pures:
            p = np.zeros(n0)
            q = np.zeros(n1)
            p[i] = 1.0
            q[j] = 1.0
            eqs.append(Equilibrium(p, q, float(A[i, j]), float(B[i, j])))
        return eqs, False
    return all_equilibria(A, B), True


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
    known = {s: i for i, s in enumerate(states)}
    n0, n1 = n_actions

    # Precompute per state: outcomes[s][a0][a1] = list of (prob, next_state_idx_or_None, r0, r1)
    outcomes: list[list[list[list[tuple[float, int | None, float, float]]]]] = []
    for s in states:
        grid = [[[] for _ in range(n1)] for _ in range(n0)]
        for a0 in range(n0):
            for a1 in range(n1):
                for prob, s_next in transition(s, a0, a1):
                    r0, r1 = reward(s, a0, a1, s_next)
                    grid[a0][a1].append((prob, known.get(s_next), r0, r1))
        outcomes.append(grid)

    def matrices(si: int, v0: np.ndarray, v1: np.ndarray):
        A = np.zeros((n0, n1))
        B = np.zeros((n0, n1))
        grid = outcomes[si]
        for a0 in range(n0):
            for a1 in range(n1):
                for prob, ni, r0, r1 in grid[a0][a1]:
                    for rew, value, payoff in ((r0, v0, A), (r1, v1, B)):
                        cont = 0.0 if ni is None else value[ni]
                        payoff[a0, a1] += prob * (rew + gamma * cont)
        return A, B

    v0 = np.zeros(len(states))
    v1 = np.zeros(len(states))

    iterations = 0
    for iterations in range(1, max_iters + 1):
        nv0 = np.zeros(len(states))
        nv1 = np.zeros(len(states))
        for si in range(len(states)):
            A, B = matrices(si, v0, v1)
            eqs, _ = _equilibria(A, B)
            eq = select_equilibrium(eqs, rule=select)
            nv0[si] = eq.row_value
            nv1[si] = eq.col_value
        delta = max(np.abs(nv0 - v0).max(), np.abs(nv1 - v1).max())
        v0, v1 = nv0, nv1
        if delta < tol:
            break

    row_policy: dict[State, np.ndarray] = {}
    col_policy: dict[State, np.ndarray] = {}
    multi: list[State] = []
    se_states: list[State] = []
    for si, s in enumerate(states):
        A, B = matrices(si, v0, v1)
        eqs, used_se = _equilibria(A, B)
        if used_se:
            se_states.append(s)
        if len(eqs) > 1:
            multi.append(s)
        eq = select_equilibrium(eqs, rule=select)
        row_policy[s] = eq.row
        col_policy[s] = eq.col

    return MarkovGameResult(
        {s: v0[i] for s, i in known.items()},
        {s: v1[i] for s, i in known.items()},
        row_policy,
        col_policy,
        iterations,
        multi,
        se_states,
    )
