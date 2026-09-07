"""Nash Q-iteration for the zero-sum soccer Markov game.

The game is two-player zero-sum, so the stage game at each state is a matrix
game and the fixed point of

    V(s) = val( E[ R(s, .,.) + gamma * V(s') ] )

is the minimax value function (Shapley 1953). The expectation is over the
(possibly stochastic) transition; the immediate reward is not discounted.
Three stage solvers are offered:

* ``"mixed"``  -- always take the LP minimax value.
* ``"pure"``   -- always take the pure maximin (security) value.
* ``"hybrid"`` -- pure saddle value when it exists, LP value otherwise.

After convergence the solver reports the states whose stage game has no pure
saddle: a pure stationary equilibrium of the whole Markov game exists iff that
set is empty.

``run()`` is value iteration -- it solves a matrix game at every state on every
sweep. ``run_policy_iteration()`` is the "freeze then iterate" scheme: solve the
stage games once, hold the strategies fixed for several cheap linear
policy-evaluation sweeps, then re-solve. It reaches the same fixed point with
far fewer (expensive) matrix-game solves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soccer_nash.game import JOINT_ACTIONS, SoccerGame, State
from soccer_nash.matrix_games import (
    game_value,
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
    matrix_game_solves: int = 0  # LP calls (0 for a purely pure/saddle run)
    staleness_trace: list[float] | None = None  # policy iteration only

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
        shaping=None,
    ):
        if mode not in {"mixed", "pure", "hybrid"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.game = game
        self.gamma = gamma
        self.mode = mode
        self.tol = tol
        self.max_iters = max_iters
        self.shaping = shaping
        self._lp_calls = 0

        self._states: list[State] = list(game.states())
        # Per state: a 4x4 grid of outcome lists [(prob, next_state, r0), ...],
        # with any shaping reward folded into r0.
        self._out: dict[State, np.ndarray] = {}
        for s in self._states:
            grid = np.empty((4, 4), dtype=object)
            for k, (a0, a1) in enumerate(JOINT_ACTIONS):
                i, j = divmod(k, 4)
                outcomes = []
                for prob, ns, (r0, _) in game.transitions(s, a0, a1):
                    if shaping is not None:
                        r0 = r0 + shaping.reward_delta(game, s, ns, gamma)
                    outcomes.append((prob, ns, r0))
                grid[i, j] = outcomes
            self._out[s] = grid

    # ------------------------------------------------------------------ stage
    def _matrix(self, s: State, values: dict[State, float]) -> np.ndarray:
        m = np.zeros((4, 4))
        grid = self._out[s]
        gamma = self.gamma
        for i in range(4):
            for j in range(4):
                acc = 0.0
                for prob, ns, r0 in grid[i, j]:
                    if self.game.is_terminal(ns):
                        acc += prob * r0
                    else:
                        acc += prob * (r0 + gamma * values[ns])
                m[i, j] = acc
        return m

    def _stage_value(self, m: np.ndarray) -> float:
        if self.mode == "pure":
            return security_strategy_row(m)[1]
        if self.mode == "mixed":
            self._lp_calls += 1
            return game_value(m)
        lo, hi = pure_bounds(m)
        if hi - lo <= _SADDLE_TOL:
            return lo
        self._lp_calls += 1
        return game_value(m)

    @staticmethod
    def _pure_strategies(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One-hot maximin row for player 0, minimax column for player 1.

        Player 0's security value for a row is that row's minimum over the
        columns (``axis=1``); player 1's for a column is that column's maximum
        over the rows (``axis=0``).
        """
        p = np.zeros(4)
        q = np.zeros(4)
        p[int(np.argmax(m.min(axis=1)))] = 1.0
        q[int(np.argmin(m.max(axis=0)))] = 1.0
        return p, q

    def _stage_policy(self, m: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Equilibrium (or security) strategies for the stage game."""
        if self.mode == "pure":
            p, q = self._pure_strategies(m)
            return p, q, False
        if self.mode == "hybrid":
            lo, hi = pure_bounds(m)
            if hi - lo <= _SADDLE_TOL:
                p, q = self._pure_strategies(m)
                return p, q, False
        self._lp_calls += 1
        _, p, q = solve_zero_sum(m)
        return p, q, True

    # -------------------------------------------------------------- iteration
    def run(self) -> NashQResult:
        """Value iteration: solve a matrix game at every state, every sweep."""
        self._lp_calls = 0
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
            matrix_game_solves=self._lp_calls,
        )

    def run_policy_iteration(
        self, eval_sweeps: int = 50, max_outer: int = 200
    ) -> NashQResult:
        """Freeze the stage-game strategies, run cheap linear evaluation sweeps,
        then re-solve. Same fixed point, far fewer matrix-game solves."""
        self._lp_calls = 0
        values: dict[State, float] = {s: 0.0 for s in self._states}
        row_policy = {s: np.full(4, 0.25) for s in self._states}
        col_policy = {s: np.full(4, 0.25) for s in self._states}
        staleness_trace: list[float] = []

        outer = 0
        for outer in range(1, max_outer + 1):
            # --- improvement: re-solve every stage game from the current V.
            # policy_delta is exactly the staleness Brandon flagged: how far the
            # frozen strategies were from the Nash of the current Q.
            policy_delta = 0.0
            for s in self._states:
                p, q, _ = self._stage_policy(self._matrix(s, values))
                policy_delta = max(
                    policy_delta,
                    float(np.abs(p - row_policy[s]).max()),
                    float(np.abs(q - col_policy[s]).max()),
                )
                row_policy[s], col_policy[s] = p, q
            staleness_trace.append(policy_delta)

            # --- evaluation: strategies frozen, linear backups only
            eval_delta = 0.0
            for _ in range(eval_sweeps):
                eval_delta = 0.0
                updated: dict[State, float] = {}
                for s in self._states:
                    m = self._matrix(s, values)
                    v = float(row_policy[s] @ m @ col_policy[s])
                    eval_delta = max(eval_delta, abs(v - values[s]))
                    updated[s] = v
                values = updated
                if eval_delta < self.tol:
                    break

            if policy_delta < 1e-9 and eval_delta < self.tol:
                break

        _, _, no_saddle = self._extract_policies(values)
        return NashQResult(
            values=values,
            row_policy=row_policy,
            col_policy=col_policy,
            no_saddle_states=no_saddle,
            iterations=outer,
            mode=self.mode,
            gamma=self.gamma,
            matrix_game_solves=self._lp_calls,
            staleness_trace=staleness_trace,
        )

    def value_bracket_gaps(self, result: "NashQResult") -> np.ndarray:
        """Per-state certified value error of the solved policy.

        For each state the row player can guarantee ``min_j (p M)_j`` and reach
        at most ``max_i (M q)_i``; the difference bounds how far the reported
        value can be from the true minimax value at that state.
        """
        from soccer_nash.numerics import value_bracket

        gaps = np.empty(len(self._states))
        for i, s in enumerate(self._states):
            m = self._matrix(s, result.values)
            gaps[i] = value_bracket(m, result.row_policy[s], result.col_policy[s]).gap
        return gaps

    def optimal_action_masks(
        self, result: "NashQResult", tol: float = 1e-6
    ) -> tuple[list[State], np.ndarray, np.ndarray]:
        """Per-state 0/1 masks of each player's security-optimal actions.

        A row is optimal for player 0 if its worst-case entry equals the
        maximin value; a column is optimal for player 1 if its best-case entry
        equals the minimax value. Where a pure saddle exists these coincide
        with the equilibrium supports.
        """
        states = list(result.values)
        row = np.zeros((len(states), 4))
        col = np.zeros((len(states), 4))
        for i, s in enumerate(states):
            m = self._matrix(s, result.values)
            # Player 0 maximises its worst case over the row; player 1 minimises
            # player 0's best case over the column.
            row_security = m.min(axis=1)
            col_security = m.max(axis=0)
            row[i] = row_security >= row_security.max() - tol  # player 0 maximises
            col[i] = col_security <= col_security.min() + tol  # player 1 minimises
        return states, row, col

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
                p, q = self._pure_strategies(m)
            else:
                no_saddle.append(s)
                _, p, q = solve_zero_sum(m)
            row_policy[s] = p
            col_policy[s] = q
        return row_policy, col_policy, no_saddle
