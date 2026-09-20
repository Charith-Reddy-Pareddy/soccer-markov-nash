import numpy as np
import pytest

from soccer_nash.game import A10SoccerGame, SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.policy_gradient import (
    PolicyNet,
    _discounted_returns,
    evaluate_policy_gradient,
    pretrain_policy_nets,
    train_reinforce_selfplay,
)


def test_discounted_returns_is_a_backward_suffix_sum():
    rewards = [1.0, 2.0, 3.0]
    g = _discounted_returns(rewards, gamma=0.5)
    assert g[2] == pytest.approx(3.0)
    assert g[1] == pytest.approx(2.0 + 0.5 * 3.0)
    assert g[0] == pytest.approx(1.0 + 0.5 * (2.0 + 0.5 * 3.0))


def test_discounted_returns_empty_is_empty():
    assert _discounted_returns([], gamma=0.9) == []


def test_policy_net_outputs_a_valid_distribution():
    net = PolicyNet(hidden=8)
    p = net.policy((0, 0, 2, 2, 0))
    assert p.shape == (4,)
    assert np.isclose(p.sum(), 1.0)
    assert (p >= 0.0).all()


def test_train_reinforce_selfplay_runs_on_a_tiny_game():
    game = A10SoccerGame(width=5, height=3, goal_rows=(1,))
    result = train_reinforce_selfplay(
        game, gamma=0.9, hidden=8, iterations=5, rollout_len=10, seed=0
    )
    assert len(result.mean_reward_trace) == 5
    p = result.net0.policy(game.initial_state())
    q = result.net1.policy(game.initial_state())
    assert np.isclose(p.sum(), 1.0)
    assert np.isclose(q.sum(), 1.0)


def test_train_reinforce_selfplay_is_seed_reproducible():
    game = A10SoccerGame(width=5, height=3, goal_rows=(1,))
    a = train_reinforce_selfplay(
        game, gamma=0.9, hidden=8, iterations=3, rollout_len=10, seed=1
    )
    b = train_reinforce_selfplay(
        game, gamma=0.9, hidden=8, iterations=3, rollout_len=10, seed=1
    )
    np.testing.assert_allclose(
        a.net0.policy(game.initial_state()), b.net0.policy(game.initial_state())
    )
    assert a.mean_reward_trace == pytest.approx(b.mean_reward_trace)


def test_evaluate_policy_gradient_against_exact_solver_is_well_formed():
    game = A10SoccerGame(width=5, height=3, goal_rows=(1,))
    exact = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9)
    solver.run()
    matrix_of = lambda s: solver._matrix(s, exact.values)  # noqa: E731

    result = train_reinforce_selfplay(
        game, gamma=0.9, hidden=8, iterations=5, rollout_len=10, seed=0
    )
    m = evaluate_policy_gradient(
        game, result, exact.row_policy, exact.col_policy, matrix_of, gamma=0.9
    )
    assert 0.0 <= m["row_action_agreement"] <= 1.0
    assert 0.0 <= m["col_action_agreement"] <= 1.0
    assert m["mean_equilibrium_regret"] >= 0.0
    assert m["max_equilibrium_regret"] >= m["mean_equilibrium_regret"]
    assert isinstance(m["duality_gap"], float)
    # a best-responder is the worst-case opponent, so it can never do worse
    # (for the net whose value is being reported) than a random one, for any
    # fixed policy -- not just a well-trained one.
    assert m["row_vs_random"] >= m["row_vs_best_response"] - 1e-6
    assert m["col_vs_random"] >= m["col_vs_best_response"] - 1e-6


def test_pretrain_policy_nets_moves_toward_the_exact_policy():
    # Not a full fit -- just that supervised regression toward the exact
    # policy actually reduces the distance from a random init, and produces
    # valid distributions.
    game = A10SoccerGame(width=5, height=3, goal_rows=(1,))
    exact = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()
    fresh = PolicyNet(hidden=8)
    net0, net1 = pretrain_policy_nets(
        game, exact.row_policy, exact.col_policy, hidden=8, epochs=60, seed=0
    )
    s0 = game.initial_state()
    p_fresh = fresh.policy(s0)
    p_fit = net0.policy(s0)
    exact_p = exact.row_policy[s0]
    assert np.abs(p_fit - exact_p).sum() < np.abs(p_fresh - exact_p).sum()
    assert np.isclose(net1.policy(s0).sum(), 1.0)


def test_train_reinforce_selfplay_init_net_is_the_actual_starting_point():
    # epochs=0-equivalent: iterations=0 skips the training loop entirely, so
    # the returned net's weights should be an exact copy of init_net's, not a
    # fresh random init -- same property nash_dqn.py's train_nash_dqn tests
    # for its own init_net.
    game = A10SoccerGame(width=5, height=3, goal_rows=(1,))
    exact = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()
    warm0, warm1 = pretrain_policy_nets(
        game, exact.row_policy, exact.col_policy, hidden=8, epochs=20, seed=0
    )
    result = train_reinforce_selfplay(
        game, gamma=0.9, hidden=8, iterations=0, rollout_len=10, seed=1,
        init_net0=warm0, init_net1=warm1,
    )
    s0 = game.initial_state()
    np.testing.assert_allclose(result.net0.policy(s0), warm0.policy(s0))
    np.testing.assert_allclose(result.net1.policy(s0), warm1.policy(s0))

    # and it must differ from what a from-scratch (seed=1) init would give --
    # otherwise this test wouldn't actually be exercising init_net at all
    from_scratch = train_reinforce_selfplay(
        game, gamma=0.9, hidden=8, iterations=0, rollout_len=10, seed=1
    )
    assert not np.allclose(result.net0.policy(s0), from_scratch.net0.policy(s0))


def test_selfplay_rewards_are_zero_sum():
    # r1 == -r0 at every step, always -- the assumption g1 = [-g for g in g0]
    # in train_reinforce_selfplay relies on.
    game = SoccerGame(width=5, height=4, goal_rows=(1, 2))
    rng = np.random.default_rng(0)
    state = game.initial_state()
    for _ in range(50):
        ns, (r0, r1), done = game.step(
            state, game.actions()[0], game.actions()[1], rng=rng
        )
        assert r0 == -r1
        state = game.initial_state() if done else ns
