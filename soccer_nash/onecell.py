"""The single-goal-cell pure-saddle certificate.

Two independently checkable facts, for the random-move-order game with exactly
one goal cell per side:

1. **The defender has an explicit safe strategy** (:func:`guard_action`): move
   onto the goal row, then along it toward the goal cell, never onto the
   carrier. Verified to hold the carrier to at most ``V*(s)`` from every state,
   which with weak duality forces ``minimax(M_s) = V*(s)`` (and, on the states
   where the defender is player 0, ``maximin(M_s) = V*(s)``).
2. **Every stage game is IEWDS-solvable** (:mod:`soccer_nash.dominance`), hence
   has a pure saddle.

:func:`certify` runs both over one board and returns the slack (0 iff the proof
goes through). ``docs/proof.md`` has the argument.
"""

from __future__ import annotations

from dataclasses import dataclass

from soccer_nash.dominance import is_iewds_solvable
from soccer_nash.game import Action, SoccerGame, State
from soccer_nash.nash_q import NashQIteration

_D = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
_SCORE = {0: Action.R, 1: Action.L}  # the carrying player's scoring action


def _carrier_defender(s: State, width: int):
    x0, y0, x1, y1, b = s
    if b == 0:                       # player 0 carries, attacks the right edge
        return (x0, y0), (x1, y1), width - 1
    return (x1, y1), (x0, y0), 0     # player 1 carries, attacks the left edge


def guard_action(game: SoccerGame, s: State) -> Action:
    """The defender's safe move for a single-goal-cell game: reach the goal
    row, then slide along it toward the goal cell, never stepping onto the
    carrier."""
    if len(game.goal_rows) != 1:
        raise ValueError("guard_action is only defined for a single goal row")
    (cx, cy), (dx, dy), tx = _carrier_defender(s, game.width)
    g = game.goal_rows[0]
    w, h = game.width, game.height

    best, best_a = None, 0
    for a in range(4):
        ddx, ddy = _D[a]
        nx = min(max(dx + ddx, 0), w - 1)
        ny = min(max(dy + ddy, 0), h - 1)
        if (nx, ny) == (cx, cy):
            key = (9, 9, 9)                       # never step onto the carrier
        else:
            key = (abs(ny - g), abs(nx - tx), -(abs(nx - cx) + abs(ny - cy)))
        if best is None or key < best:
            best, best_a = key, a
    return Action(best_a)


@dataclass
class Certificate:
    width: int
    height: int
    gamma: float
    states: int
    guard_slack: float          # max amount the carrier beats V* against the guard
    iewds_failures: int         # stage games not solved by weak-dominance elimination
    kickoff_value: float
    undiscounted_mixed_stage_games: int  # over the full backward-induction horizon

    @property
    def ok(self) -> bool:
        return (
            self.guard_slack < 1e-6
            and self.iewds_failures == 0
            and self.undiscounted_mixed_stage_games == 0
        )


def certify(width: int, height: int, gamma: float = 0.9) -> Certificate:
    """Check the single-cell pure-saddle proof on one board.

    Three checks: the defender's guard strategy secures ``V*`` (stationary,
    ``gamma``); every stationary stage game is weak-dominance solvable; and the
    *undiscounted* game (exact backward induction) has no mixed stage game at
    any horizon.
    """
    game = SoccerGame(
        width=width, height=height, goal_rows=(height // 2,), move_order="random"
    )
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-12)
    result = solver.run()
    V = result.values
    fh = solver.run_finite_horizon()

    def cont(ns: State) -> float:
        return 0.0 if game.is_terminal(ns) else gamma * V[ns]

    guard_slack = 0.0
    iewds_failures = 0
    for s in solver._states:
        b = s[4]
        beta = int(guard_action(game, s))
        for a_carrier in range(4):
            a0, a1 = (a_carrier, beta) if b == 0 else (beta, a_carrier)
            q = sum(
                p * (r0 + cont(ns))
                for p, ns, (r0, _r1) in game.transitions(s, Action(a0), Action(a1))
            )
            carrier_gain = (q - V[s]) if b == 0 else (V[s] - q)
            guard_slack = max(guard_slack, carrier_gain)

        if not is_iewds_solvable(solver._matrix(s, V)):
            iewds_failures += 1

    return Certificate(
        width, height, gamma, len(solver._states),
        guard_slack, iewds_failures, V[game.initial_state()],
        fh.mixed_stage_games,
    )
