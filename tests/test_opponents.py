import pytest

from soccer_nash.game import Action, SoccerGame
from soccer_nash.opponents import always_left, part2_opponent


@pytest.fixture
def game():
    return SoccerGame()


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def test_always_left(game):
    for s in list(game.states())[::113]:
        assert always_left(game, s, 0) == Action.L


def test_chaser_closes_on_the_carrier(game):
    # Opponent is player 1, does NOT have the ball -> chases player 0.
    s = (2, 3, 5, 1, 0)
    a = part2_opponent(game, s, me=1)
    assert a == Action.L  # horizontal first: 5 -> 4 toward x = 2


def test_chaser_uses_vertical_when_aligned(game):
    s = (5, 4, 5, 1, 0)  # same column, carrier above
    assert part2_opponent(game, s, me=1) == Action.U


def test_carrier_heads_for_its_goal(game):
    # Player 1 has the ball; its goal is the left edge.
    assert part2_opponent(game, (3, 2, 4, 2, 1), me=1) == Action.L
    # Player 0 has the ball; its goal is the right edge.
    assert part2_opponent(game, (3, 2, 4, 2, 0), me=0) == Action.R


def test_carrier_squares_up_to_goal_row_before_scoring(game):
    # Player 1 with ball at the left edge but off the goal rows -> move onto one.
    assert part2_opponent(game, (4, 2, 0, 0, 1), me=1) == Action.U
    # On a goal row at the edge -> the scoring move.
    assert part2_opponent(game, (4, 3, 0, 2, 1), me=1) == Action.L


def test_returns_a_legal_action_everywhere(game):
    for s in game.states():
        for me in (0, 1):
            assert part2_opponent(game, s, me) in set(Action)


def test_chaser_move_reduces_distance(game):
    for s in list(game.states())[::71]:
        x0, y0, x1, y1, b = s
        if b == 0:  # player 1 (me=1) has no ball -> should chase player 0
            me_pos, tgt = (x1, y1), (x0, y0)
            a = part2_opponent(game, s, me=1)
            nxt, _, _ = game.step(s, Action.U, a)  # player 0 action irrelevant here
            # recompute me's new position directly from the action
            dx, dy = {Action.U: (0, 1), Action.D: (0, -1),
                      Action.L: (-1, 0), Action.R: (1, 0)}[a]
            new_me = (me_pos[0] + dx, me_pos[1] + dy)
            assert _manhattan(new_me, tgt) <= _manhattan(me_pos, tgt)
