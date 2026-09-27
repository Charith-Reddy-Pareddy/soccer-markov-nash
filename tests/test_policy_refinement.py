"""Regressions for weakly dominated security ties in the explorer case."""

import numpy as np
import pytest

from soccer_nash.game import Action, SoccerGame
from soccer_nash.matrix_games import security_strategy_row
from soccer_nash.nash_q import NashQIteration

REPORTED_Q = np.array([
    [0, 0, 0, 0],
    [.729, 0, .729, 0],
    [0, 0, 0, 0],
    [.81, .81, -.59049, 0],
])


def test_reported_state_prefers_down_without_sacrificing_security():
    p, q = NashQIteration._pure_strategies(REPORTED_Q)
    assert p == pytest.approx([0, 1, 0, 0])
    assert q == pytest.approx([0, 0, 0, 1])
    assert min(p @ REPORTED_Q) == max(REPORTED_Q @ q) == 0


def test_security_tie_refinement_is_player_symmetric():
    p, q = NashQIteration._pure_strategies(-REPORTED_Q.T)
    assert p == pytest.approx([0, 0, 0, 1])
    assert q == pytest.approx([0, 1, 0, 0])


def test_refinement_does_not_trade_security_for_average_payoff():
    i, value = security_strategy_row(np.array([[0, 0], [-1e-12, 100]]))
    assert i == 0
    assert value == 0


def test_refinement_handles_rectangular_games_and_identical_actions():
    p, q = NashQIteration._pure_strategies(np.array([[0, 0, 0], [0, 2, 1]]))
    assert p == pytest.approx([0, 1])
    assert q == pytest.approx([1, 0, 0])
    p, q = NashQIteration._pure_strategies(np.zeros((2, 3)))
    assert p == pytest.approx([1, 0])
    assert q == pytest.approx([1, 0, 0])


@pytest.mark.parametrize('a0,a1,expected', [
    ('U', 'U', (4, 4, 5, 4, 0)), ('U', 'D', (4, 4, 5, 2, 0)),
    ('U', 'L', (4, 4, 4, 3, 0)), ('U', 'R', (4, 4, 6, 3, 0)),
    ('D', 'U', (4, 2, 5, 4, 0)), ('D', 'D', (4, 2, 5, 2, 0)),
    ('D', 'L', (4, 2, 4, 3, 0)), ('D', 'R', (4, 2, 6, 3, 0)),
    ('L', 'U', (3, 3, 5, 4, 0)), ('L', 'D', (3, 3, 5, 2, 0)),
    ('L', 'L', (3, 3, 4, 3, 0)), ('L', 'R', (3, 3, 6, 3, 0)),
    ('R', 'U', (5, 3, 5, 4, 0)), ('R', 'D', (5, 3, 5, 2, 0)),
    ('R', 'L', (5, 3, 4, 3, 1)), ('R', 'R', (5, 3, 6, 3, 0)),
])
def test_reported_state_hand_calculated_transitions(a0, a1, expected):
    game = SoccerGame()
    assert game.transitions((4, 3, 5, 3, 0), Action[a0], Action[a1]) == [
        (1.0, expected, (0, 0))
    ]
    assert expected[:2] != expected[2:4]
