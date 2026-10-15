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


def test_pretrained_critic_fits_exact_values_for_both_player_perspectives(small):
    import torch

    game, solver, *_ = small
    exact = fh.ExactCritic(game, solver, GAMMA, HORIZON)
    states = list(exact.states)
    x = torch.cat([pf.features(states, t, HORIZON) for t in range(HORIZON)])
    ids = torch.tensor([exact.code(*s) for _ in range(HORIZON) for s in states])
    times = torch.arange(HORIZON).repeat_interleave(len(states))
    target = exact.V[times, ids]
    torch.manual_seed(7)
    c0, c1 = pf.Net(game, 1), pf.Net(game, 1)
    before = torch.nn.functional.mse_loss(c0(x).squeeze(-1), target)
    pf.fit_exact_critic(c0, exact, 0, epochs=80, batch_size=len(x))
    pf.fit_exact_critic(c1, exact, 1, epochs=80, batch_size=len(x))
    after = torch.nn.functional.mse_loss(c0(x).squeeze(-1), target)
    assert after < before * 0.5
    assert torch.nn.functional.mse_loss(c1(x).squeeze(-1), -target) < before
    assert all(not p.requires_grad for p in c0.parameters())
    assert all(not p.requires_grad for p in c1.parameters())


def test_pretrained_critic_is_frozen_during_policy_updates_and_runs_both_modes(small):
    import torch

    game, solver, *_ = small
    exact = fh.ExactCritic(game, solver, GAMMA, HORIZON)
    net, critic = pf.Net(game, 4), pf.Net(game, 1)
    pf.fit_exact_critic(critic, exact, 0, epochs=2, batch_size=128)
    batch = pf.collect(game, pf._Current(net), pf._Current(pf.Net(game, 4)), 4, HORIZON,
                       np.random.default_rng(11))
    opts = [torch.optim.Adam([*net.parameters(), *critic.parameters()]) for _ in range(2)]
    before = [p.detach().clone() for p in critic.parameters()]
    exact.action_value = lambda _: pytest.fail("pretrained actor update queried exact action values")
    pf.update(net, critic, opts, batch, 0, "a2c_pretrained", GAMMA, 0.0, exact=exact)
    assert all(torch.equal(p.detach(), old) for p, old in zip(critic.parameters(), before))
    assert all(p.grad is None for p in critic.parameters())

    for mode in ("selfplay", "fictitious"):
        tr = pf.train(game, "a2c_pretrained", mode, GAMMA, HORIZON, iterations=2,
                      episodes=4, snap_every=1, exact=exact,
                      critic_pretrain_epochs=2, critic_pretrain_batch_size=128)
        assert tr.critics is not None and len(tr.critics) == 2
        assert all(not p.requires_grad for c in tr.critics for p in c.parameters())
        assert tr.pol0(0, [game.initial_state()]).sum() == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError):
        pf.train(game, "a2c_pretrained", "selfplay", GAMMA, HORIZON,
                 iterations=1, episodes=2)


def test_pretrained_critic_supports_random_move_order_board(small):
    game = SoccerGame(width=4, height=3, goal_rows=(1,), move_order="random",
                      max_steps=HORIZON)
    solver = NashQIteration(game, gamma=GAMMA, mode="hybrid", tol=1e-10)
    exact = fh.ExactCritic(game, solver, GAMMA, HORIZON)
    tr = pf.train(game, "a2c_pretrained", "selfplay", GAMMA, HORIZON,
                  iterations=1, episodes=3, exact=exact, seed=2,
                  critic_pretrain_epochs=2, critic_pretrain_batch_size=128)
    assert tr.pol1(0, [game.initial_state()]).sum() == pytest.approx(1.0, abs=1e-5)


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
    assert args.seeds == 3 and args.entropy == 0.01 and args.updates_per_batch == 5
    pretrained_args = mod.parser().parse_args(["--algos", "a2c_pretrained"])
    assert pretrained_args.algos == ["a2c_pretrained"]
    one_update = mod.parser().parse_args(["--updates-per-batch", "1"])
    assert one_update.updates_per_batch == 1


def test_training_forwards_multiple_updates_for_each_collected_player_batch(small, monkeypatch):
    game = small[0]
    seen = []
    update = pf.update

    def record(*args, **kwargs):
        seen.append((kwargs["updates_per_batch"], id(args[3]["X"])))
        return update(*args, **kwargs)

    monkeypatch.setattr(pf, "update", record)
    pf.train(game, "a2c", "selfplay", GAMMA, HORIZON, iterations=1, episodes=3,
             updates_per_batch=2, seed=5)
    assert [n for n, _ in seen] == [2, 2]
    assert seen[0][1] == seen[1][1]
    with pytest.raises(ValueError, match="updates_per_batch"):
        pf.train(game, "a2c", "selfplay", GAMMA, HORIZON, iterations=1,
                 episodes=2, updates_per_batch=0)


def test_resumed_runs_are_read_back_as_numbers_so_the_final_summary_can_average_them(tmp_path):
    import importlib.util
    import pathlib
    import statistics

    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "pg_finite.py"
    spec = importlib.util.spec_from_file_location("pg_finite_script2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = tmp_path / "out.csv"
    f.write_text("algo,mode,seed,train_s,row_win_vs_br\n"
                 "exact,-,0,0.0,0.5\na2c,fictitious,1,3020.4,0.25\na2c,fictitious,2,3183.3,0.75\n")
    rows = mod.load_runs(f)
    assert [r["seed"] for r in rows] == [1, 2] and rows[0]["mode"] == "fictitious"
    assert statistics.mean(r["row_win_vs_br"] for r in rows) == 0.5


def test_minibatches_take_several_gradient_steps_where_full_batch_takes_one(small):
    import torch

    game = small[0]
    batch = pf.collect(game, pf._Current(pf.Net(game, 4)), pf._Current(pf.Net(game, 4)), 8,
                       HORIZON, np.random.default_rng(1))

    def steps(minibatch):
        net, count = pf.Net(game, 4), [0]
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        real = opt.step
        opt.step = lambda *a, **k: (count.__setitem__(0, count[0] + 1), real(*a, **k))[1]
        pf.update(net, None, [opt, opt], batch, 0, "reinforce", GAMMA, 0.01, minibatch=minibatch)
        return count[0]

    n = int(batch["alive"].sum())
    assert steps(0) == 1 and steps(16) == -(-n // 16)


@pytest.mark.parametrize("algo", ["reinforce", "a2c", "a2c_exact", "a2c_pretrained"])
def test_configured_updates_reuse_one_collected_batch_and_ppo_keeps_its_epochs(small, algo):
    import torch

    game, solver, *_ = small
    exact = fh.ExactCritic(game, solver, GAMMA, HORIZON) if algo in pf.EXACT_ALGOS else None
    batch = pf.collect(game, pf._Current(pf.Net(game, 4)), pf._Current(pf.Net(game, 4)), 4,
                       HORIZON, np.random.default_rng(13))
    net = pf.Net(game, 4)
    critic = pf.Net(game, 1) if algo in ("a2c", "a2c_pretrained") else None
    if algo == "a2c_pretrained":
        pf.fit_exact_critic(critic, exact, 0, epochs=1, batch_size=128)
    params = [*net.parameters(), *(critic.parameters() if critic is not None else [])]
    opt = torch.optim.SGD(params, lr=0.01)
    calls = [0]
    step = opt.step
    opt.step = lambda *a, **k: (calls.__setitem__(0, calls[0] + 1), step(*a, **k))[1]
    pf.update(net, critic, [opt, opt], batch, 0, algo, GAMMA, 0.01, exact=exact,
              updates_per_batch=3)
    assert calls[0] == 3

    ppo_net = pf.Net(game, 4)
    ppo_critic = pf.Net(game, 1)
    ppo_opt = torch.optim.SGD([*ppo_net.parameters(), *ppo_critic.parameters()], lr=0.01)
    ppo_calls = [0]
    ppo_step = ppo_opt.step
    ppo_opt.step = lambda *a, **k: (ppo_calls.__setitem__(0, ppo_calls[0] + 1),
                                    ppo_step(*a, **k))[1]
    pf.update(ppo_net, ppo_critic, [ppo_opt, ppo_opt], batch, 0, "ppo", GAMMA, 0.01,
              updates_per_batch=3)
    assert ppo_calls[0] == 4


def test_replay_concatenates_episodes_and_every_algorithm_trains_with_it(small):
    game = small[0]
    one = pf.collect(game, pf._Current(pf.Net(game, 4)), pf._Current(pf.Net(game, 4)), 3,
                     HORIZON, np.random.default_rng(0))
    both = pf._concat([one, one])
    assert both["X"].shape[1] == 6 and both["A"].shape[2] == 6
    assert both["alive"].shape == (HORIZON, 6)
    assert pf._concat([one]) is one
    for algo in pf.ALGOS:
        tr = pf.train(game, algo, "selfplay", GAMMA, HORIZON, iterations=3, episodes=4,
                      replay=2, minibatch=8)
        assert tr.pol0(0, [game.initial_state()]).sum() == pytest.approx(1.0, abs=1e-5)


def test_checkpoints_report_both_the_aggregate_and_the_current_policy(small):
    game = small[0]
    seen = []
    pf.train(game, "reinforce", "fictitious", GAMMA, HORIZON, iterations=4, episodes=4,
             snap_every=1, checkpoint_every=2,
             on_checkpoint=lambda it, rep, cur: seen.append((it, rep, cur)))
    assert [s[0] for s in seen] == [2, 4]
    start = [game.initial_state()]
    agg, cur = seen[-1][1].pol0(0, start), seen[-1][2].pol0(0, start)
    assert agg.shape == cur.shape == (1, 4) and agg.sum() == pytest.approx(1.0, abs=1e-5)


def test_weight_averaging_of_identical_snapshots_is_the_snapshot_and_differs_otherwise(small):
    game = small[0]
    net = pf.Net(game, 4)
    start = [game.initial_state()]
    same = pf.weight_average([net, pf._frozen(net)], HORIZON)(0, start)
    assert np.allclose(same, pf._policy([net], HORIZON)(0, start), atol=1e-6)
    other = pf.Net(game, 4)
    mixed = pf.weight_average([net, other], HORIZON)(0, start)
    last = pf.weight_average([net, other], HORIZON, last_layer_only=True)(0, start)
    assert not np.allclose(mixed, last, atol=1e-6) and mixed.sum() == pytest.approx(1.0, abs=1e-5)


def test_snapshots_are_returned_for_the_fictitious_modes_only(small):
    game = small[0]
    fic = pf.train(game, "reinforce", "fictitious", GAMMA, HORIZON, iterations=4, episodes=4,
                   snap_every=2)
    assert fic.snaps is not None and len(fic.snaps[0]) == 3
    assert pf.train(game, "reinforce", "selfplay", GAMMA, HORIZON, iterations=2,
                    episodes=4).snaps is None


def test_the_exact_critic_also_trains_on_a_game_that_ends_at_the_first_goal(small):
    game, solver = small[0], small[1]
    critic = fh.ExactCritic(game, solver, GAMMA, HORIZON)
    for mode in ("selfplay", "fictitious"):
        tr = pf.train(game, "a2c_exact", mode, GAMMA, HORIZON, iterations=3, episodes=4,
                      snap_every=1, exact=critic)
        assert tr.pol0(0, [game.initial_state()]).sum() == pytest.approx(1.0, abs=1e-5)
