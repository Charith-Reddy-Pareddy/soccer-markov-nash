"""Scripted opponent policies for the soccer game.

* ``handbuilt_policy`` -- the deterministic "simple rules for scoring and
  blocking" opponent from Littman (1994) §6.2. On offense it drives straight for
  the goal it attacks; on defense it moves to the square in front of the carrier
  (between the carrier and the goal it defends) and **stands** there, since
  "a good defensive maneuver is to stand where the other player wants to go"
  (§6.1). Needs the ``stand`` action to block properly (``n_actions=5``).
* ``part2_opponent`` -- a simpler chaser: head to goal with the ball, tail the
  carrier without it. Kept for the A10 Part 2 deliverables.

Both resolve the horizontal direction before the vertical one.
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


def handbuilt_policy(game: SoccerGame, state: State, me: int) -> Action:
    """Littman (1994) §6.2: deterministic, "simple rules for scoring and
    blocking"."""
    x0, y0, x1, y1, b = state
    my = (x0, y0) if me == 0 else (x1, y1)
    carrier = (x1, y1) if me == 0 else (x0, y0)
    forward = 1 if me == 0 else -1                 # my attacking x-direction

    if b == me:
        # Scoring: get onto a goal row, then drive at the attacking edge.
        goal_y = _nearest_goal_row(my[1], game.goal_rows)
        if my[1] != goal_y:
            return Action.U if goal_y > my[1] else Action.D
        goal_x = game.width - 1 if me == 0 else 0
        if my[0] != goal_x:
            return Action.R if goal_x > my[0] else Action.L
        return Action.R if me == 0 else Action.L      # step off the edge -> score

    # Blocking: predict the carrier's next square (it runs the same offense
    # toward the goal I defend) and stand on it -- "stand where the other player
    # wants to go".
    goal_x = 0 if me == 0 else game.width - 1        # the goal I defend
    goal_y = _nearest_goal_row(carrier[1], game.goal_rows)
    if carrier[1] != goal_y:
        want = (carrier[0], carrier[1] + (1 if goal_y > carrier[1] else -1))
    else:
        want = (carrier[0] - forward, carrier[1])
    want = (min(max(want[0], 0), game.width - 1),
            min(max(want[1], 0), game.height - 1))
    if my == want:
        return Action.STAND if game.n_actions == 5 else _step_toward(my, carrier)
    return _step_toward(my, want)


def always_left(game: SoccerGame, state: State, me: int) -> Action:
    """The A10 Part 1 Q6 opponent: always play ``L``."""
    return Action.L
