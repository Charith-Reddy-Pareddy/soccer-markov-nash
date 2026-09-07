import numpy as np
import pytest

from soccer_nash.game import A10SoccerGame, SoccerGame
from soccer_nash.nash_dqn import _minimax, compare_to_exact, train_nash_dqn
from soccer_nash.nash_q import NashQIteration


def test_minimax_matches_the_lp_on_a_random_matrix():
    from soccer_nash.matrix_games import solve_zero_sum

    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.normal(size=(4, 4))
        assert _minimax(m) == pytest.approx(solve_zero_sum(m)[0], abs=1e-7)


def test_rejects_stochastic_game():
    with pytest.raises(ValueError):
        train_nash_dqn(SoccerGame(move_order="random"))


def test_qnet_can_regress_toward_the_exact_stage_matrices():
    # Not a full training run -- just that the loss decreases and the comparison
    # metrics are well-formed.
    game = A10SoccerGame(width=5, height=3, goal_rows=(1,))
    exact = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()
    result = train_nash_dqn(game, gamma=0.9, hidden=48, epochs=40, seed=0)
    assert result.loss_trace[-1] < result.loss_trace[0]

    m = compare_to_exact(game, result.net, exact.values, exact.row_policy, 0.9)
    assert 0.0 <= m["action_agreement"] <= 1.0
    assert m["max_value_error"] >= 0.0
    assert m["duality_gap"] >= -1e-9
