"""Scripted opponent policies for the soccer game.

The A10 Part 2 opponent:

* if it does **not** have the ball, it moves toward the ball carrier;
* if it **has** the ball, it moves toward its own goal;

always resolving the horizontal direction before the vertical one.
"""

from __future__ import annotations

from soccer_nash.game import Action, SoccerGame, State


def _step_toward(
    pos: tuple[int, int], target: tuple[int, int]
) -> Action:
    """One move from ``pos`` toward ``target``, horizontal first."""
    (x, y), (tx, ty) = pos, target
    if x < tx:
        return Action.R
    if x > tx:
        return Action.L
    if y < ty:
        return Action.U
    if y > ty:
        return Action.D
    # Already on the target column and row: nudge horizontally (used by the
    # ball carrier standing in the goal mouth -- this is the scoring move).
    return Action.L if tx <= x else Action.R


def _nearest_goal_row(y: int, goal_rows: tuple[int, ...]) -> int:
    if y in goal_rows:
        return y
    return min(goal_rows, key=lambda g: abs(g - y))


def part2_opponent(game: SoccerGame, state: State, me: int) -> Action:
    """Action for player ``me`` under the A10 Part 2 scripted policy."""
    x0, y0, x1, y1, b = state
    my_pos = (x0, y0) if me == 0 else (x1, y1)
    their_pos = (x1, y1) if me == 0 else (x0, y0)

    if b == me:
        # Head for the goal: column 0 (player 1) or width-1 (player 0), on the
        # nearest goal row.
        goal_x = 0 if me == 1 else game.width - 1
        goal_y = _nearest_goal_row(my_pos[1], game.goal_rows)
        return _step_toward(my_pos, (goal_x, goal_y))

    # Chase the ball carrier.
    return _step_toward(my_pos, their_pos)


def always_left(game: SoccerGame, state: State, me: int) -> Action:
    """The A10 Part 1 Q6 opponent: always play ``L``."""
    return Action.L
