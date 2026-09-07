"""The dog / debate game verifies the general-sum solver against a closed form:
the only Nash equilibrium parks each player in its own house, with the dog at
the weighted average of the two houses.
"""

import numpy as np
import pytest

from soccer_nash.dog_game import DogGame
from soccer_nash.markov_game import solve_markov_game


@pytest.fixture(scope="module")
def solved():
    game = DogGame(size=4, w_a=0.5)
    result = solve_markov_game(
        game.states(), game.n_actions, game.transition, game.reward,
        gamma=0.9, tol=1e-6, max_iters=400,
    )
    return game, result


def test_house_b_defaults_to_the_far_corner():
    assert DogGame(size=5).house_b == (4, 4)
    assert DogGame(size=5, line=True).house_b == (4, 0)  # stays on the line


def test_dog_sits_at_the_weighted_midpoint():
    assert DogGame(size=4, w_a=0.5).equilibrium_dog() == (1.5, 1.5)
    assert DogGame(size=5, w_a=0.75).equilibrium_dog() == (1.0, 1.0)


def test_players_park_in_their_corners_at_equilibrium(solved):
    game, result = solved
    corners = (game.house_a, game.house_b)
    assert np.argmax(result.row_policy[corners]) == 0  # "stay"
    assert np.argmax(result.col_policy[corners]) == 0


def test_rollout_converges_to_the_closed_form(solved):
    game, result = solved
    s = ((2, 2), (0, 0))
    for _ in range(40):
        a = int(np.argmax(result.row_policy[s]))
        b = int(np.argmax(result.col_policy[s]))
        s = game.transition(s, a, b)[0][1]
    assert np.allclose(game.dog(*s), game.equilibrium_dog(), atol=1e-6)


def test_one_dimensional_variant_also_hits_the_closed_form():
    game = DogGame(size=5, line=True)
    result = solve_markov_game(
        game.states(), game.n_actions, game.transition, game.reward,
        gamma=0.9, tol=1e-7, max_iters=400,
    )
    eq = (game.house_a, game.house_b)
    assert np.argmax(result.row_policy[eq]) == 0
    assert game.dog(*eq) == pytest.approx(game.equilibrium_dog())


def test_dog_game_is_solved_purely(solved):
    _, result = solved
    # Every stage game of the dog game has a pure Nash equilibrium.
    assert result.support_enum_states == []


def test_asymmetric_weights_pull_the_dog():
    game = DogGame(size=4, w_a=0.25)
    result = solve_markov_game(
        game.states(), game.n_actions, game.transition, game.reward,
        gamma=0.9, tol=1e-6, max_iters=400,
    )
    s = ((2, 2), (1, 1))
    for _ in range(40):
        a = int(np.argmax(result.row_policy[s]))
        b = int(np.argmax(result.col_policy[s]))
        s = game.transition(s, a, b)[0][1]
    # w_a = 0.25 -> dog closer to house_b = (3, 3): 0.25*0 + 0.75*3 = 2.25
    assert game.dog(*s) == pytest.approx((2.25, 2.25))
