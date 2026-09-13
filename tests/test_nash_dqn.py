import numpy as np
import pytest

from soccer_nash.game import A10SoccerGame, SoccerGame
from soccer_nash.nash_dqn import (
    _minimax,
    _QNet,
    compare_policy_to_exact,
    compare_to_exact,
    fit_q_to_exact,
    train_nash_dqn,
    train_policy_baseline,
)
from soccer_nash.nash_q import NashQIteration


def test_minimax_matches_the_lp_on_a_random_matrix():
    from soccer_nash.matrix_games import solve_zero_sum

    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.normal(size=(4, 4))
        assert _minimax(m) == pytest.approx(solve_zero_sum(m)[0], abs=1e-7)


def test_qnet_int_hidden_matches_the_equivalent_tuple():
    # h=64 is documented to mean "2 hidden layers of width 64" -- same as
    # passing (64, 64) explicitly. Same seed should give bit-identical
    # weights either way, since this is the width/depth ablation's whole
    # premise: the int shorthand is not a different architecture.
    a = _QNet(64, seed=3, out=16)
    b = _QNet((64, 64), seed=3, out=16)
    for wa, wb in zip(a._w(), b._w()):
        np.testing.assert_array_equal(wa, wb)


def test_qnet_supports_variable_depth_and_width():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 5))
    target = rng.normal(size=(12, 16))
    for hidden in [(32,), (48, 48, 48), (16, 16, 16, 16)]:
        net = _QNet(hidden, seed=0, out=16)
        assert len(net.Ws) == len(hidden) + 1
        assert net.Ws[0].shape == (5, hidden[0])
        assert net.Ws[-1].shape == (hidden[-1], 16)
        loss0 = net.step(X, target)
        for _ in range(20):
            loss = net.step(X, target)
        assert loss < loss0  # actually learns something over a few steps


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

    m = compare_to_exact(
        game, result.net, exact.values, exact.row_policy, 0.9,
        exact_no_saddle=set(exact.no_saddle_states),
    )
    assert 0.0 <= m["action_agreement"] <= 1.0
    assert 0.0 <= m["classification_agreement"] <= 1.0
    assert m["max_value_error"] >= m["mean_value_error"] >= 0.0
    assert m["duality_gap"] >= -1e-9


def test_fit_q_to_exact_regresses_toward_the_exact_stage_matrices():
    game = A10SoccerGame(width=5, height=3, goal_rows=(1,))
    exact = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9)
    solver.run()
    matrix_of = lambda s: solver._matrix(s, exact.values)  # noqa: E731

    net = fit_q_to_exact(game, matrix_of, hidden=48, epochs=60, seed=0)
    m = compare_to_exact(
        game, net, exact.values, exact.row_policy, 0.9,
        exact_no_saddle=set(exact.no_saddle_states),
    )
    # supervised regression onto the exact matrices, no bootstrap at all --
    # should fit noticeably closer than a handful of TD-bootstrap epochs would
    assert m["max_value_error"] < 1.0
    assert 0.0 <= m["action_agreement"] <= 1.0
    assert m["duality_gap"] >= -1e-9


def test_train_nash_dqn_init_net_is_the_actual_starting_point():
    # epochs=0 skips the training loop entirely, so the returned net's
    # weights should be an exact copy of init_net's, not a fresh random init.
    game = A10SoccerGame(width=5, height=3, goal_rows=(1,))
    exact = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9)
    solver.run()
    matrix_of = lambda s: solver._matrix(s, exact.values)  # noqa: E731

    warm_start = fit_q_to_exact(game, matrix_of, hidden=48, epochs=20, seed=0)
    result = train_nash_dqn(
        game, gamma=0.9, hidden=48, epochs=0, seed=1, init_net=warm_start,
    )
    s0 = game.initial_state()
    np.testing.assert_allclose(result.net.matrix(s0), warm_start.matrix(s0))

    # and it must differ from what a from-scratch (seed=1) init would give --
    # otherwise this test wouldn't actually be exercising init_net at all
    from_scratch = train_nash_dqn(game, gamma=0.9, hidden=48, epochs=0, seed=1)
    assert not np.allclose(result.net.matrix(s0), from_scratch.net.matrix(s0))


def test_policy_baseline_regresses_toward_the_exact_strategies():
    game = A10SoccerGame(width=5, height=3, goal_rows=(1,))
    exact = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9)
    solver.run()
    matrix_of = lambda s: solver._matrix(s, exact.values)  # noqa: E731

    net = train_policy_baseline(
        game, exact.row_policy, exact.col_policy, hidden=48, epochs=60, seed=0
    )
    m = compare_policy_to_exact(
        game, net, matrix_of, exact.row_policy,
        set(exact.no_saddle_states), 0.9,
    )
    # trained on the exact strategies, it should name the right action often
    assert m["action_agreement"] > 0.7
    assert m["max_equilibrium_regret"] >= 0.0
    assert m["duality_gap"] >= -1e-9
    p, q = net.policy(game.initial_state())
    assert np.isclose(p.sum(), 1.0) and np.isclose(q.sum(), 1.0)
