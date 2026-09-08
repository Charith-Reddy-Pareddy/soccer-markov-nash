"""Reachability attractors and the pure memoryless equilibrium."""

import pytest

from soccer_nash.attractor import (
    verify_positional_equilibrium,
    win_attractor,
)
from soccer_nash.game import SoccerGame


def test_win_attractor_rejects_stochastic_games():
    with pytest.raises(ValueError):
        win_attractor(SoccerGame(move_order="random"), 0)


def test_scoring_state_is_rank_zero():
    game = SoccerGame(width=5, height=3, goal_rows=(1,))
    a0 = win_attractor(game, 0)
    # player 0 on the goal cell with the ball, defender parked away -> wins now
    assert a0[(4, 1, 0, 0, 0)] == 0


def test_attractors_are_mirror_images_in_size():
    game = SoccerGame(width=5, height=3, goal_rows=(1,))
    assert len(win_attractor(game, 0)) == len(win_attractor(game, 1))


@pytest.mark.parametrize(
    ("w", "h", "goal_rows"),
    [(5, 3, (1,)), (3, 5, (2,)), (5, 3, (0, 1, 2)), (5, 5, (1, 2, 3))],
)
def test_positional_profile_realizes_the_value(w, h, goal_rows):
    game = SoccerGame(width=w, height=h, goal_rows=goal_rows,
                      move_order="deterministic")
    c = verify_positional_equilibrium(game)
    assert c.ok
    assert c.a0 == c.a1
    assert c.a0 + c.a1 + c.draw == c.states


@pytest.mark.slow
def test_positional_profile_on_the_full_board():
    game = SoccerGame(width=7, height=5, goal_rows=(1, 2, 3),
                      move_order="deterministic")
    assert verify_positional_equilibrium(game).ok
