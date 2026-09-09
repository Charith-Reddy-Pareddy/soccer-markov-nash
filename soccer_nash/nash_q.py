"""Nash Q-iteration for the zero-sum soccer Markov game.

The game is two-player zero-sum, so the stage game at each state is a matrix
game and the fixed point of

    V(s) = val( E[ R(s, .,.) + gamma * V(s') ] )

is the minimax value function (Shapley 1953). The expectation is over the
(possibly stochastic) transition; the immediate reward is not discounted.
Three stage backups are offered:

* ``"mixed"``  -- the LP minimax value ``val(M) = max_p min_j (p^T M)_j``. This
  is the true Nash value of the (zero-sum) stage game.
* ``"pure"``   -- the **pure-security / maximin** value ``max_i min_j M[i,j]``.
  This quantity always exists, but when it is strictly below the minimax it is
  *not* a Nash equilibrium value -- it is what the row player can guarantee by
  committing to a single action. Use it as a fast lower bound, not as a solver.
* ``"hybrid"`` -- the pure *saddle* value when ``maximin == minimax`` (then it
  equals the Nash value), the LP value otherwise. Exact everywhere.

After convergence the solver reports the states whose converged stage game has
no pure saddle. If that set is empty, playing a pure saddle action at every
state is a stationary pure-strategy equilibrium of the Markov game (see
``docs/assumptions.md``, Claim B).

``run()`` is stationary value iteration with ``gamma < 1`` -- a matrix game at
every state on every sweep. ``run_finite_horizon()`` solves the *undiscounted*
game (the actual A10 game: +/-1 at a goal, tie after ``max_steps``) exactly by
backward induction. ``run_policy_iteration()`` is the "freeze then iterate"
scheme: solve the stage games once, hold the strategies fixed for several cheap
linear policy-evaluation sweeps, then re-solve -- the same fixed point with far
fewer matrix-game solves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soccer_nash.game import SoccerGame, State
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
    residual_trace: list[float] | None = None  # max Bellman update per sweep

    @property
    def pure_equilibrium_exists(self) -> bool:
        return not self.no_saddle_states

    @property
    def saddle_fraction(self) -> float:
        total = len(self.values)
        return 1.0 - len(self.no_saddle_states) / total if total else 1.0


@dataclass
class FiniteHorizonResult:
    """Backward induction for the undiscounted, horizon-capped game."""

    values: dict[State, float]        # V with the full horizon remaining
    no_saddle_states: list[State]     # states whose stage game is mixed at some step
    mixed_stage_games: int            # total (state, step) pairs with no pure saddle
    horizon: int

    @property
    def pure_equilibrium_exists(self) -> bool:
        return not self.no_saddle_states


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
        self._lp_cache: dict = {}

        self._n = game.n_actions
        self._joint = game.joint_actions()
        self._states: list[State] = list(game.states())
        # Per state: an n x n grid of outcome lists [(prob, next_state, r0), ...],
        # with any shaping reward folded into r0.
        self._out: dict[State, np.ndarray] = {}
        for s in self._states:
            grid = np.empty((self._n, self._n), dtype=object)
            for k, (a0, a1) in enumerate(self._joint):
                i, j = divmod(k, self._n)
                outcomes = []
                for prob, ns, (r0, _) in game.transitions(s, a0, a1):
                    if shaping is not None:
                        r0 = r0 + shaping.reward_delta(game, s, ns, gamma)
                    outcomes.append((prob, ns, r0))
                grid[i, j] = outcomes
            self._out[s] = grid

    # ------------------------------------------------------------------ stage
    def _matrix(
        self, s: State, values: dict[State, float], gamma: float | None = None
    ) -> np.ndarray:
        m = np.zeros((self._n, self._n))
        grid = self._out[s]
        gamma = self.gamma if gamma is None else gamma
        for i in range(self._n):
            for j in range(self._n):
                acc = 0.0
                for prob, ns, r0 in grid[i, j]:
                    if self.game.is_terminal(ns):
                        acc += prob * r0
                    else:
                        acc += prob * (r0 + gamma * values[ns])
                m[i, j] = acc
        return m

    def _cached_game_value(self, m: np.ndarray) -> float:
        """LP minimax value, memoised on the (rounded) matrix content. Near
        convergence most stage matrices repeat sweep to sweep, so this turns
        thousands of LP calls into a few hundred."""
        key = np.round(m, 11).tobytes()
        hit = self._lp_cache.get(key)
        if hit is not None:
            return hit
        self._lp_calls += 1
        v = game_value(m)
        self._lp_cache[key] = v
        return v

    def _stage_value(self, m: np.ndarray) -> float:
        if self.mode == "pure":
            return security_strategy_row(m)[1]
        if self.mode == "mixed":
            return self._cached_game_value(m)
        lo, hi = pure_bounds(m)
        if hi - lo <= _SADDLE_TOL:
            return lo
        return self._cached_game_value(m)

    @staticmethod
    def _pure_strategies(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One-hot maximin row for player 0, minimax column for player 1.

        Player 0's security value for a row is that row's minimum over the
        columns (``axis=1``); player 1's for a column is that column's maximum
        over the rows (``axis=0``).
        """
        p = np.zeros(m.shape[0])
        q = np.zeros(m.shape[1])
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
        self._lp_cache: dict = {}
        values: dict[State, float] = dict.fromkeys(self._states, 0.0)

        iterations = 0
        residual_trace: list[float] = []
        for iterations in range(1, self.max_iters + 1):
            delta = 0.0
            updated: dict[State, float] = {}
            for s in self._states:
                nv = self._stage_value(self._matrix(s, values))
                delta = max(delta, abs(nv - values[s]))
                updated[s] = nv
            values = updated
            residual_trace.append(delta)
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
            residual_trace=residual_trace,
        )

    def run_finite_horizon(self, horizon: int | None = None) -> FiniteHorizonResult:
        """Solve the *undiscounted* game exactly by backward induction.

        The A10 game is undiscounted -- `+1` / `-1` at a goal, `0` otherwise, a
        tie after ``game.max_steps`` steps -- so its exact solution is backward
        induction from `V = 0` at the horizon with `gamma = 1`. The value
        function is non-stationary (it depends on the steps remaining); this
        returns `V` with the full horizon left, the states whose stage game
        lacks a pure saddle at *some* step, and the total number of such
        (state, step) pairs. ``run()`` with `gamma < 1` is a faster stationary
        approximation to the same qualitative answer.
        """
        self._lp_calls = 0
        self._lp_cache = {}
        h = self.game.max_steps if horizon is None else horizon
        values: dict[State, float] = dict.fromkeys(self._states, 0.0)
        ever_mixed: set[State] = set()
        mixed_stage_games = 0

        for _ in range(h):
            updated: dict[State, float] = {}
            for s in self._states:
                m = self._matrix(s, values, gamma=1.0)
                lo, hi = pure_bounds(m)
                if hi - lo > _SADDLE_TOL:
                    ever_mixed.add(s)
                    mixed_stage_games += 1
                    updated[s] = self._cached_game_value(m)
                else:
                    updated[s] = 0.5 * (lo + hi)
            values = updated

        return FiniteHorizonResult(
            values=values,
            no_saddle_states=sorted(ever_mixed),
            mixed_stage_games=mixed_stage_games,
            horizon=h,
        )

    def run_symmetric(self) -> NashQResult:
        """Value iteration that solves one state per mirror pair and
        reconstructs the other by ``V(mirror(s)) = -V(s)``. Same fixed point,
        half the stage-game work. Deterministic move order only -- the mirror
        of a stochastic transition is not verified here."""
        if self.game.move_order != "deterministic":
            raise ValueError("run_symmetric() requires deterministic move order")
        if self._n != 4:
            raise ValueError("run_symmetric() is only wired for the 4-action game")
        if self.game.scoring != "win":
            raise ValueError("run_symmetric() requires scoring='win'")
        from soccer_nash.symmetry import canonical_pairs

        self._lp_calls = 0
        self._lp_cache: dict = {}
        reps, image_of = canonical_pairs(self.game)
        rep_set = set(reps)

        def val(s: State, table: dict[State, float]) -> float:
            if self.game.is_terminal(s):
                return 0.0
            return table[s] if s in rep_set else -table[image_of[s]]

        values: dict[State, float] = dict.fromkeys(reps, 0.0)
        iterations = 0
        for iterations in range(1, self.max_iters + 1):
            delta = 0.0
            updated: dict[State, float] = {}
            for s in reps:
                grid = self._out[s]
                m = np.zeros((self._n, self._n))
                for i in range(self._n):
                    for j in range(self._n):
                        acc = 0.0
                        for prob, ns, r0 in grid[i, j]:
                            acc += prob * (
                                r0
                                if self.game.is_terminal(ns)
                                else r0 + self.gamma * val(ns, values)
                            )
                        m[i, j] = acc
                nv = self._stage_value(m)
                delta = max(delta, abs(nv - values[s]))
                updated[s] = nv
            values = updated
            if delta < self.tol:
                break

        # Expand to the full state set, then extract policies as usual.
        full = {s: (values[s] if s in rep_set else -values[image_of[s]]) for s in self._states}
        row_policy, col_policy, no_saddle = self._extract_policies(full)
        return NashQResult(
            values=full,
            row_policy=row_policy,
            col_policy=col_policy,
            no_saddle_states=no_saddle,
            iterations=iterations,
            mode=self.mode,
            gamma=self.gamma,
            matrix_game_solves=self._lp_calls,
        )

    def run_policy_iteration(
        self, eval_sweeps: int = 50, max_outer: int = 200,
        eval_value: str = "mid",
    ) -> NashQResult:
        """Freeze the stage-game strategies, run cheap linear evaluation sweeps,
        then re-solve. Same fixed point, far fewer matrix-game solves.

        ``eval_value`` picks which of the three stage-game quantities the
        evaluation sweep backs up while the frozen ``(p, q)`` are still stale --
        the choice the professor flagged as an open question:

        * ``"mid"`` -- ``p M q`` (what both get if both mix). Unbiased at the
          equilibrium, noisy while the strategies are wrong.
        * ``"lower"`` -- ``min_j (p M)_j`` (what the row player *guarantees*
          against any column). A pessimistic contraction: converges from below.
        * ``"upper"`` -- ``max_i (M q)_i`` (what the row player could reach if it
          best-responded to ``q``). Optimistic: converges from above.
        """
        if eval_value not in ("mid", "lower", "upper"):
            raise ValueError(f"unknown eval_value {eval_value!r}")
        self._lp_calls = 0
        self._lp_cache: dict = {}
        values: dict[State, float] = dict.fromkeys(self._states, 0.0)
        row_policy = {s: np.full(self._n, 1 / self._n) for s in self._states}
        col_policy = {s: np.full(self._n, 1 / self._n) for s in self._states}
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
                    p, q = row_policy[s], col_policy[s]
                    if eval_value == "mid":
                        v = float(p @ m @ q)
                    elif eval_value == "lower":
                        v = float((p @ m).min())
                    else:  # upper
                        v = float((m @ q).max())
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

    def value_bracket_gaps(self, result: NashQResult) -> np.ndarray:
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
        self, result: NashQResult, tol: float = 1e-6
    ) -> tuple[list[State], np.ndarray, np.ndarray]:
        """Per-state 0/1 masks of each player's security-optimal actions.

        A row is optimal for player 0 if its worst-case entry equals the
        maximin value; a column is optimal for player 1 if its best-case entry
        equals the minimax value. Where a pure saddle exists these coincide
        with the equilibrium supports.
        """
        states = list(result.values)
        row = np.zeros((len(states), self._n))
        col = np.zeros((len(states), self._n))
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
