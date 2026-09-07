import pytest

from soccer_nash.a10 import (
    ACTION_LABELS,
    format_rewards,
    format_successors,
    rewards,
    successors,
)
from soccer_nash.game import SoccerGame


@pytest.fixture
def game():
    return SoccerGame()


def test_action_labels_match_a10_order():
    assert ACTION_LABELS == [
        "UU", "UD", "UL", "UR",
        "DU", "DD", "DL", "DR",
        "LU", "LD", "LL", "LR",
        "RU", "RD", "RL", "RR",
    ]


def test_successors_has_16_entries_of_5_ints(game):
    succ = successors(game, (3, 2, 4, 2, 0))
    assert len(succ) == 16
    assert all(len(s) == 5 for s in succ)


def test_successors_match_step(game):
    from soccer_nash.game import JOINT_ACTIONS

    state = (2, 3, 5, 1, 1)
    succ = successors(game, state)
    for (a0, a1), s in zip(JOINT_ACTIONS, succ):
        assert game.step(state, a0, a1)[0] == s


def test_scoring_joint_actions_are_terminal(game):
    state = (6, 2, 5, 3, 0)  # carrier on the goal row at the right edge
    succ = successors(game, state)
    rew = rewards(game, state, player=0)
    for label, s, r in zip(ACTION_LABELS, succ, rew):
        if label[0] == "R":  # player 0 moves into the goal
            assert s == (-1, -1, -1, -1, 0)
            assert r == 1
        else:
            assert not game.is_terminal(s)
            assert r == 0


def test_rewards_are_zero_sum(game):
    state = (0, 2, 1, 2, 1)  # player 1 next to its own goal
    r0 = rewards(game, state, player=0)
    r1 = rewards(game, state, player=1)
    assert all(a == -b for a, b in zip(r0, r1))
    assert set(r0) <= {-1, 0, 1}


def test_format_shapes(game):
    state = (3, 2, 4, 2, 0)
    lines = format_successors(game, state).splitlines()
    assert len(lines) == 16
    assert all(len(line.split(",")) == 5 for line in lines)

    reward_line = format_rewards(game, state)
    assert len(reward_line.split(",")) == 16


def test_random_move_order_rejected():
    with pytest.raises(ValueError):
        successors(SoccerGame(move_order="random"), (3, 2, 4, 2, 0))
