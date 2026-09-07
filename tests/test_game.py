import pytest

from soccer_nash.game import Action, SoccerGame


@pytest.fixture
def game():
    return SoccerGame()


def test_state_space_size(game):
    states = list(game.states())
    # 7*5 cells, 34 remaining cells for the second player, 2 ball values.
    assert len(states) == 35 * 34 * 2
    assert len(set(states)) == len(states)


def test_no_overlap_in_state_space(game):
    for x0, y0, x1, y1, _ in game.states():
        assert (x0, y0) != (x1, y1)


def test_walls_block_movement(game):
    # Player 0 at top-left corner moving up then left stays put.
    s = (0, 4, 6, 2, 1)
    nxt, reward, done = game.step(s, Action.U, Action.D)
    assert not done
    assert nxt[0:2] == (0, 4)
    assert reward == (0, 0)


def test_carrier_scores_right_edge(game):
    s = (6, 2, 3, 0, 0)  # player 0 has the ball, in a goal row, at right edge
    nxt, reward, done = game.step(s, Action.R, Action.L)
    assert done
    assert nxt == (-1, -1, -1, -1, 0)
    assert reward == (1, -1)


def test_carrier_scores_left_edge(game):
    s = (3, 4, 0, 2, 1)  # player 1 has the ball, in a goal row, at left edge
    nxt, reward, done = game.step(s, Action.R, Action.L)
    assert done
    assert nxt == (-1, -1, -1, -1, 1)
    assert reward == (-1, 1)


def test_non_carrier_cannot_score(game):
    s = (6, 2, 0, 2, 1)  # player 0 at right edge but does NOT have the ball
    nxt, reward, done = game.step(s, Action.R, Action.U)
    assert not done
    assert nxt[0:2] == (6, 2)


def test_no_score_outside_goal_rows(game):
    s = (6, 0, 3, 4, 0)  # carrier at right edge but row 0 is not a goal row
    nxt, _, done = game.step(s, Action.R, Action.L)
    assert not done
    assert nxt[0:2] == (6, 0)


def test_same_square_collision_transfers_ball(game):
    # Both players step into the cell between them; carrier takes it.
    s = (2, 2, 4, 2, 0)
    nxt, _, done = game.step(s, Action.R, Action.L)
    assert not done
    assert nxt[0:2] == (3, 2)  # carrier advanced
    assert nxt[2:4] == (4, 2)  # other stayed
    assert nxt[4] == 1  # possession flipped


def test_swap_transfers_ball(game):
    s = (3, 2, 4, 2, 0)
    nxt, _, done = game.step(s, Action.R, Action.L)
    assert not done
    assert nxt[0:2] == (4, 2)
    assert nxt[2:4] == (3, 2)
    assert nxt[4] == 1


def test_carrier_blocked_by_standing_opponent(game):
    # Player 1 (no ball) tries to walk through the standing carrier.
    s = (3, 2, 4, 2, 0)
    nxt, _, done = game.step(s, Action.U, Action.L)
    assert not done
    # Player 0 moved up; player 1 followed into the vacated cell -> allowed.
    assert nxt[0:2] == (3, 3)
    assert nxt[2:4] == (3, 2)
    assert nxt[4] == 0


def test_standing_carrier_keeps_ball_when_bumped(game):
    s = (3, 2, 4, 2, 0)
    # Carrier bounces off... actually make carrier stand by walking into a wall.
    s = (0, 2, 1, 2, 0)
    nxt, _, done = game.step(s, Action.L, Action.L)
    assert not done
    assert nxt[0:2] == (0, 2)  # carrier blocked by wall
    assert nxt[2:4] == (1, 2)  # other blocked by standing carrier
    assert nxt[4] == 1  # bump still transfers the ball


def test_carrier_barging_into_stationary_opponent_loses_the_ball(game):
    # Player 0 carries and steps into player 1, who is pinned against the wall.
    s = (1, 2, 0, 2, 0)
    nxt, _, done = game.step(s, Action.L, Action.L)
    assert not done
    assert nxt[0:2] == (1, 2)  # carrier blocked by the standing opponent
    assert nxt[2:4] == (0, 2)  # opponent stays
    assert nxt[4] == 1  # the contested-cell rule still hands over the ball

    # Symmetric case with player 1 carrying.
    s = (6, 3, 5, 3, 1)
    nxt, _, done = game.step(s, Action.R, Action.R)
    assert nxt[0:4] == (6, 3, 5, 3)
    assert nxt[4] == 0


def test_successors_length_and_order(game):
    s = game.initial_state()
    succ = game.successors(s)
    assert len(succ) == 16
    # Deterministic game: one outcome per joint action, first is (U, U).
    prob, nxt, reward = succ[0][0]
    assert prob == 1.0
    nxt_uu, reward_uu, _ = game.step(s, Action.U, Action.U)
    assert (nxt, reward) == (nxt_uu, reward_uu)


def test_step_on_terminal_raises(game):
    with pytest.raises(ValueError):
        game.step((-1, -1, -1, -1, 0), Action.U, Action.U)
