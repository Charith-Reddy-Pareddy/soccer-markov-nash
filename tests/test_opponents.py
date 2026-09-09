import pytest

from soccer_nash.game import Action, SoccerGame
from soccer_nash.opponents import always_left, handbuilt_policy, part2_opponent


@pytest.fixture
def game():
    return SoccerGame()


@pytest.fixture
def littman():
    return SoccerGame(width=5, height=4, goal_rows=(1, 2),
                      move_order="random", n_actions=5, scoring="rate")


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


def test_handbuilt_scores_off_the_edge(littman):
    # player 0 with the ball on a goal row at its goal column -> step off the edge
    assert handbuilt_policy(littman, (4, 2, 1, 2, 0), me=0) == Action.R
    # player 1 symmetric
    assert handbuilt_policy(littman, (3, 2, 0, 1, 1), me=1) == Action.L


def test_handbuilt_gets_onto_a_goal_row_first(littman):
    # player 0 with the ball off the goal rows -> move toward one (row 0 -> up)
    assert handbuilt_policy(littman, (2, 0, 4, 2, 0), me=0) == Action.U


def test_handbuilt_stands_on_the_carriers_forward_cell(littman):
    # player 1 defends; carrier (player 0) at (2,2) wants (3,2); defender there
    a = handbuilt_policy(littman, (2, 2, 3, 2, 0), me=1)
    assert a == Action.STAND


def test_handbuilt_is_legal_everywhere(littman):
    for s in list(littman.states())[::97]:
        for me in (0, 1):
            assert handbuilt_policy(littman, s, me) in set(littman.actions())


def test_handbuilt_beats_a_random_opponent(littman):
    from soccer_nash.best_response import materialize
    from soccer_nash.evaluate import policy_value
    from soccer_nash.exploit import onehot_policy, uniform_policy

    hb = onehot_policy(materialize(littman, handbuilt_policy, 0), 5)
    v = policy_value(littman, hb, uniform_policy(littman), gamma=0.9)
    assert v[littman.initial_state()] > 0.4      # a competent bot, not a stooge


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
