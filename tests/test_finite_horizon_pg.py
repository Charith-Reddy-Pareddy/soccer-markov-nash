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


# ---- the continuing game (play restarts after a goal) and the exact critic ----------------------
@pytest.fixture(scope="module")
def continuing():
    game = SoccerGame(width=4, height=3, goal_rows=(1,), move_order="deterministic",
                      scoring="rate", max_steps=HORIZON)
    solver = NashQIteration(game, gamma=GAMMA, mode="hybrid", tol=1e-10)
    return game, solver


def test_in_the_continuing_game_no_game_ends_early_and_the_rates_sum_to_one(continuing):
    game, _ = continuing
    last_step_players = []

    def recording(t, states):
        if t == HORIZON - 1:
            last_step_players.append(len(states))
        return fh.uniform(t, states)

    r = fh.play(game, recording, fh.uniform, 30, HORIZON, seed=3)
    assert last_step_players == [30]            # every game was still running at the last step
    assert r["win"] + r["tie"] + r["loss"] == pytest.approx(1.0)


def test_the_exact_critic_advantage_is_q_minus_v_of_the_exact_solution(continuing):
    game, solver = continuing
    critic = fh.ExactCritic(game, solver, GAMMA, HORIZON)
    values, _, _ = fh.solve_finite_horizon(solver, GAMMA, HORIZON)
    rng = np.random.default_rng(0)
    batch = pf.collect(game, pf._Current(pf.Net(game, 4)), pf._Current(pf.Net(game, 4)),
                       4, HORIZON, rng)
    adv = critic.advantage(batch)
    for t, e in [(0, 0), (2, 1), (HORIZON - 1, 3)]:
        st = tuple(int(v) for v in batch["X"][t, e, :5].round().tolist())
        a0, a1 = int(batch["A"][0][t, e]), int(batch["A"][1][t, e])
        want = solver._matrix(st, values[t + 1], gamma=GAMMA)[a0, a1] - values[t][st]
        assert float(adv[t, e]) == pytest.approx(want, abs=1e-5)


def test_a2c_with_the_exact_critic_trains_in_both_schemes_and_needs_the_critic(continuing):
    game, solver = continuing
    critic = fh.ExactCritic(game, solver, GAMMA, HORIZON)
    for mode in ("selfplay", "fictitious"):
        tr = pf.train(game, "a2c_exact", mode, GAMMA, HORIZON, iterations=2, episodes=4,
                      snap_every=1, exact=critic)
        p = tr.pol0(0, [game.initial_state()])
        assert p.shape == (1, 4) and p.sum() == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError):
        pf.train(game, "a2c_exact", "selfplay", GAMMA, HORIZON, iterations=1, episodes=2)


def test_shared_network_players_use_one_trunk_and_separate_output_slices(small):
    game = small[0]
    base = pf.Net(game, 8)
    h0, h1 = pf.Head(base, 0), pf.Head(base, 1)
    x = pf.features([game.initial_state()], 0, HORIZON)
    assert h0.base is h1.base
    assert np.allclose(torch_cat(h0(x), h1(x)), base(x).detach().numpy())
    for algo in pf.ALGOS:
        for mode in ("selfplay", "fictitious"):
            tr = pf.train(game, algo, mode, GAMMA, HORIZON, iterations=2, episodes=4,
                          snap_every=1, shared=True)
            p = tr.pol1(0, [game.initial_state()])
            assert p.shape == (1, 4) and p.sum() == pytest.approx(1.0, abs=1e-5)


def torch_cat(a, b):
    import torch
    return torch.cat([a, b], dim=-1).detach().numpy()


def test_trimming_every_step_gives_no_update_and_trimming_some_changes_it(small):
    import torch
    game = small[0]

    def run(trim):
        torch.manual_seed(0)
        net, critic = pf.Net(game, 4), pf.Net(game, 1)
        opts = [torch.optim.SGD([*net.parameters(), *critic.parameters()], lr=1.0)] * 2
        batch = pf.collect(game, pf._Current(net), pf._Current(pf.Net(game, 4)), 8, HORIZON,
                           np.random.default_rng(1))
        before = [q.detach().clone() for q in net.parameters()]
        pf.update(net, critic, opts, batch, 0, "a2c", GAMMA, 0.0, trim=trim)
        return max(float((q.detach() - b).abs().max())
                   for q, b in zip(net.parameters(), before))

    assert run(HORIZON) == 0.0
    assert run(0) > 0.0 and run(0) != run(2)


def test_pure_snapshots_play_their_most_likely_move_and_average_to_frequencies(small):
    import torch
    game = small[0]
    x = pf.features([game.initial_state()] * 5, 0, HORIZON)
    nets = [pf.Net(game, 4) for _ in range(3)]
    for n in nets:
        probs = pf._probs(n, x, True)
        assert torch.equal(probs.sum(-1), torch.ones(5))
        assert torch.equal(probs.argmax(-1), n(x).argmax(-1))
    avg = pf._policy(nets, HORIZON, True)(0, [game.initial_state()])
    top = [int(n(x[:1]).argmax()) for n in nets]
    expect = np.bincount(top, minlength=4) / 3
    assert np.allclose(avg[0], expect, atol=1e-6)
    mix = pf._Mixture(nets, 6, np.random.default_rng(0), pure=True)
    out = mix.probs(x[:1].repeat(6, 1), np.arange(6))
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_the_argmax_fictitious_mode_trains_with_every_algorithm_and_rejects_unknown_modes(small):
    game = small[0]
    for algo in pf.ALGOS:
        tr = pf.train(game, algo, "fictitious_argmax", GAMMA, HORIZON, iterations=3, episodes=4,
                      snap_every=1)
        p = tr.pol0(0, [game.initial_state()])
        assert p.shape == (1, 4) and p.sum() == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError):
        pf.train(game, "ppo", "argmax", GAMMA, HORIZON, iterations=1, episodes=2)


def test_the_runner_trains_for_the_documented_2000_iterations_by_default():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "pg_finite.py"
    spec = importlib.util.spec_from_file_location("pg_finite_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parser().parse_args([])
    assert args.iterations == 2000 and args.episodes == 64 and args.games == 1000
    assert args.seeds == 3 and args.entropy == 0.01
