"""Independent reward convention and full-board Bellman checks."""

import numpy as np
import pytest

from scripts.reward_q_audit import CASES, SCREENSHOT, PossessionProgressGame
from soccer_nash.game import MOVE_ACTIONS, SoccerGame
from soccer_nash.nash_q import NashQIteration


@pytest.fixture(scope="module")
def solved():
    result = []
    for cls in (SoccerGame, PossessionProgressGame):
        game = cls(width=7, height=5, goal_rows=(1, 2, 3), move_order="deterministic")
        solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-11)
        result.append((game, solver, solver.run_exact()))
    return result


def test_jae_screenshot_reproduces_with_next_state_rewards(solved):
    _, solver, result = solved[1]
    np.testing.assert_allclose(
        solver._matrix(CASES[2], result.values), SCREENSHOT, atol=5e-7, rtol=0
    )


def test_dense_rewards_use_next_carrier_and_skip_terminal_bonus():
    game = PossessionProgressGame(move_order="deterministic")
    U, D, L, R = MOVE_ACTIONS
    assert game.transitions((3, 4, 4, 4, 0), U, U)[0][2] == (0.02, -0.02)
    assert game.transitions((3, 4, 4, 4, 0), U, L)[0][2] == (-0.015, 0.015)
    assert game.transitions((6, 1, 4, 1, 0), R, U)[0][2] == (1.0, -1.0)
    assert game.transitions((4, 1, 0, 1, 1), U, L)[0][2] == (-1.0, 1.0)


def test_every_state_has_an_equilibrium_bellman_certificate(solved):
    for game, solver, result in solved:
        for s in game.states():
            matrix = solver._matrix(s, result.values)
            lo = (result.row_policy[s] @ matrix).min()
            hi = (matrix @ result.col_policy[s]).max()
            assert abs(lo - hi) < 1e-8
            assert abs(result.values[s] - lo) < 1e-8


def test_screenshot_sparse_matrix_is_not_all_zero(solved):
    _, solver, result = solved[0]
    matrix = solver._matrix(CASES[2], result.values)
    assert (matrix == 0).sum() == 14
    assert matrix[1, 2] == pytest.approx(0.6561)
    assert matrix[3, 2] == pytest.approx(-0.59049)
    # A zero-value state is not the same as an all-zero matrix.
    assert result.values[CASES[2]] == 0
    assert (solver._matrix(CASES[7], result.values) == 0).all()
