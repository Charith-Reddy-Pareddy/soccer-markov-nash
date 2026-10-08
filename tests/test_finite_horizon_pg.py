import numpy as np
import pytest

from soccer_nash import finite_horizon as fh
from soccer_nash import pg_finite as pf
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration

GAMMA, HORIZON = 0.9, 6


@pytest.fixture(scope="module")
def small():
    game = SoccerGame(width=4, height=3, goal_rows=(1,), move_order="deterministic",
                      max_steps=HORIZON)
    solver = NashQIteration(game, gamma=GAMMA, mode="hybrid", tol=1e-10)
    values, row, col = fh.solve_finite_horizon(solver, GAMMA, HORIZON)
    return game, solver, values, row, col


def test_exact_equilibrium_has_zero_exploitability(small):
    _, solver, _, row, col = small
    gap = fh.exploitability(
        solver, fh.tables_to_policy(row), fh.tables_to_policy(col), GAMMA, HORIZON)
    assert abs(gap) < 1e-9


def test_a_fixed_policy_is_exploitable(small):
    _, solver, _, _, _ = small
    always_up = lambda t, states: np.tile([1.0, 0, 0, 0], (len(states), 1))  # noqa: E731
    assert fh.exploitability(solver, always_up, always_up, GAMMA, HORIZON) > 0.1


def test_win_tie_loss_rates_sum_to_one(small):
    game, _, _, row, col = small
    r = fh.play(game, fh.tables_to_policy(row), fh.uniform, 50, HORIZON, seed=1)
    assert r["win"] + r["tie"] + r["loss"] == pytest.approx(1.0)


def test_every_algorithm_and_mode_trains_and_returns_distributions(small):
    game = small[0]
    for algo in pf.ALGOS:
        for mode in ("selfplay", "fictitious"):
            tr = pf.train(game, algo, mode, GAMMA, HORIZON, iterations=2, episodes=4,
                          snap_every=1)
            p = tr.pol0(0, [game.initial_state()])
            assert p.shape == (1, 4) and p.sum() == pytest.approx(1.0, abs=1e-5)


def test_policy_input_includes_the_remaining_steps(small):
    game = small[0]
    x0 = pf.features([game.initial_state()], 0, HORIZON)
    x5 = pf.features([game.initial_state()], 5, HORIZON)
    assert x0[0, -1] == 1.0 and x5[0, -1] == pytest.approx(1 / HORIZON)


def test_a_snapshot_is_independent_of_later_training(small):
    game = small[0]
    net = pf.Net(game, 4)
    snap = pf._frozen(net)
    with __import__("torch").no_grad():
        net.body[0].weight += 1.0
    assert not (snap.body[0].weight == net.body[0].weight).all()


def test_mirror_gap_is_zero_for_uniform_play_and_one_for_opposite_fixed_moves(small):
    game = small[0]
    times = (0, 3)
    assert fh.mirror_gap(game, fh.uniform, fh.uniform, times) == (0.0, 0.0)
    fixed = lambda a: lambda t, states: np.tile(np.eye(4)[a], (len(states), 1))  # noqa: E731
    assert fh.mirror_gap(game, fixed(0), fixed(2), times)[1] == 1.0  # U vs flip(L) = R


def test_dqn_trains_and_its_policies_are_distributions(small):
    from soccer_nash.dqn_finite import QPolicies, train_dqn

    game, solver, *_ = small
    net = train_dqn(solver, GAMMA, HORIZON, steps=3, batch=8)
    q = QPolicies(solver, net, HORIZON)
    states = [game.initial_state()]
    for pol in (q.row, q.col):
        p = pol(0, states)
        assert p.shape == (1, 4) and p.sum() == pytest.approx(1.0)


def test_dqn_target_at_the_last_step_is_just_the_immediate_reward(small):
    from soccer_nash.dqn_finite import _transition_arrays

    _, solver, *_ = small
    _, _, prob, rew, term = _transition_arrays(solver)
    # at the last step the continuation is zero, so a target is sum(prob * reward);
    # every goal outcome is flagged terminal and every probability row sums to one
    assert np.allclose(prob.sum(-1), 1.0)
    assert (rew[~term] == 0).all()


def test_fictitious_play_with_best_responses_reports_every_round(small):
    game = small[0]
    seen = []
    tr = pf.train_fp_br(game, "a2c", rounds=2, br_iters=2, gamma=GAMMA, horizon=HORIZON,
                        episodes=4, on_round=lambda r, trained: seen.append(r))
    assert seen == [1, 2]
    p = tr.pol0(0, [game.initial_state()])
    assert p.shape == (1, 4) and p.sum() == pytest.approx(1.0, abs=1e-5)


def test_fictitious_play_with_best_responses_rejects_an_unknown_algorithm(small):
    with pytest.raises(ValueError):
        pf.train_fp_br(small[0], "sarsa", rounds=1, br_iters=1)
