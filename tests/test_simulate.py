import numpy as np
import pytest

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.simulate import play_game, win_rates


@pytest.fixture(scope="module")
def solved():
    game = SoccerGame()
    result = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()
    return game, result


def test_carrier_scores_and_is_recorded_as_winner(solved):
    game, result = solved
    start = (5, 2, 0, 0, 0)  # player 0 two steps from an open goal
    r = play_game(game, result.row_policy, result.col_policy,
                  np.random.default_rng(1), start=start)
    assert r.winner == 0
    assert r.outcome_for_row == 1.0
    assert r.trajectory[-1] == (-1, -1, -1, -1, 0)


def test_defender_wins_is_recorded(solved):
    game, result = solved
    start = (6, 4, 1, 2, 1)  # player 1 two steps from its goal, carrier clear
    r = play_game(game, result.row_policy, result.col_policy,
                  np.random.default_rng(1), start=start)
    assert r.winner == 1
    assert r.outcome_for_row == -1.0


def test_stalemate_runs_full_horizon(solved):
    game, result = solved
    start = (1, 2, 5, 2, 0)  # defender parked between carrier and goal -> tie
    r = play_game(game, result.row_policy, result.col_policy,
                  np.random.default_rng(2), start=start)
    assert r.winner is None
    assert r.steps == game.max_steps


def test_win_rates_sum_to_one(solved):
    game, result = solved
    rates = win_rates(game, result.row_policy, result.col_policy, n_games=50,
                      start=(5, 2, 0, 0, 0))
    assert rates["row_win"] + rates["tie"] + rates["row_loss"] == pytest.approx(1.0)
    assert rates["row_win"] == 1.0
