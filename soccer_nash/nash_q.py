"""Nash Q-iteration for the zero-sum soccer Markov game.

Because the game is two-player zero-sum with deterministic transitions, the
stage game at each state is a matrix game and the fixed point of

    V(s) = val( R(s, .,.) + gamma * V(succ(s, .,.)) )

is the minimax value function (Shapley 1953). Three stage solvers are offered:

* ``"mixed"``  -- always take the LP minimax value.
* ``"pure"``   -- always take the pure maximin (security) value.
* ``"hybrid"`` -- pure saddle value when it exists, LP value otherwise.

After convergence the solver reports the states whose stage game has no pure
saddle: a pure stationary equilibrium of the whole Markov game exists iff that
set is empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soccer_nash.game import JOINT_ACTIONS, SoccerGame, State
from soccer_nash.matrix_games import (
    pure_bounds,
    security_strategy_row,
    solve_zero_sum,
)

_SADDLE_TOL = 1e-7


@dataclass
class NashQResult:
    values: dict[State, float]
    row_policy: dict[State, np.ndarray]
    col_policy: dict[State, np.ndarray]
    no_saddle_states: list[State]
    iterations: int
    mode: str
    gamma: float

    @property
    def pure_equilibrium_exists(self) -> bool:
        return not self.no_saddle_states

    @property
    def saddle_fraction(self) -> float:
        total = len(self.values)
        return 1.0 - len(self.no_saddle_states) / total if total else 1.0


class NashQIteration:
    def __init__(
        self,
        game: SoccerGame,
        gamma: float = 0.9,
        mode: str = "hybrid",
        tol: float = 1e-8,
        max_iters: int = 2000,
    ):
        if mode not in {"mixed", "pure", "hybrid"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.game = game
        self.gamma = gamma
        self.mode = mode
        self.tol = tol
        self.max_iters = max_iters

        self._states: list[State] = list(game.states())
        self._next: dict[State, np.ndarray] = {}
        self._reward: dict[State, np.ndarray] = {}
        for s in self._states:
            nxt = np.empty((4, 4), dtype=object)
            rew = np.zeros((4, 4))
            for k, (a0, a1) in enumerate(JOINT_ACTIONS):
                i, j = divmod(k, 4)
                ns, (r0, _), _ = game.step(s, a0, a1)
                nxt[i, j] = ns
                rew[i, j] = r0
            self._next[s] = nxt
            self._reward[s] = rew

    # ------------------------------------------------------------------ stage
    def _matrix(self, s: State, values: dict[State, float]) -> np.ndarray:
        cont = np.zeros((4, 4))
        nxt = self._next[s]
        for i in range(4):
            for j in range(4):
                ns = nxt[i, j]
                if not self.game.is_terminal(ns):
                    cont[i, j] = values[ns]
        return self._reward[s] + self.gamma * cont

    def _stage_value(self, m: np.ndarray) -> float:
        if self.mode == "pure":
            return security_strategy_row(m)[1]
        if self.mode == "mixed":
            return solve_zero_sum(m)[0]
        lo, hi = pure_bounds(m)
        if hi - lo <= _SADDLE_TOL:
            return lo
        return solve_zero_sum(m)[0]

    # -------------------------------------------------------------- iteration
    def run(self) -> NashQResult:
        values: dict[State, float] = {s: 0.0 for s in self._states}

        iterations = 0
        for iterations in range(1, self.max_iters + 1):
            delta = 0.0
            updated: dict[State, float] = {}
            for s in self._states:
                nv = self._stage_value(self._matrix(s, values))
                delta = max(delta, abs(nv - values[s]))
                updated[s] = nv
            values = updated
            if delta < self.tol:
                break

        row_policy, col_policy, no_saddle = self._extract_policies(values)
        return NashQResult(
            values=values,
            row_policy=row_policy,
            col_policy=col_policy,
            no_saddle_states=no_saddle,
            iterations=iterations,
            mode=self.mode,
            gamma=self.gamma,
        )

    def _extract_policies(
        self, values: dict[State, float]
    ) -> tuple[dict[State, np.ndarray], dict[State, np.ndarray], list[State]]:
        row_policy: dict[State, np.ndarray] = {}
        col_policy: dict[State, np.ndarray] = {}
        no_saddle: list[State] = []
        for s in self._states:
            m = self._matrix(s, values)
            lo, hi = pure_bounds(m)
            if hi - lo <= _SADDLE_TOL:
                i = int(np.argmax(m.min(axis=1)))
                j = int(np.argmin(m.max(axis=0)))
                p = np.zeros(4)
                q = np.zeros(4)
                p[i] = 1.0
                q[j] = 1.0
            else:
                no_saddle.append(s)
                _, p, q = solve_zero_sum(m)
            row_policy[s] = p
            col_policy[s] = q
        return row_policy, col_policy, no_saddle
