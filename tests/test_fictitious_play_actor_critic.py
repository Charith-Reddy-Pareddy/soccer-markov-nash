import numpy as np

from soccer_nash.actor_critic import _advantages, train_actor_critic_selfplay
from soccer_nash.fictitious_play import best_response_dynamics, fictitious_play
from soccer_nash.game import A10SoccerGame

RPS = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)


def test_best_response_dynamics_cycles_on_rps():
    path = best_response_dynamics(RPS, 30)
    assert len(set(path)) == 3 and path[:3] == path[3:6]


def test_fictitious_play_converges_on_rps():
    p, q, lo, hi = fictitious_play(RPS, 5000)
    assert np.allclose(p, 1 / 3, atol=0.01) and np.allclose(q, 1 / 3, atol=0.01)
    assert lo <= 0 <= hi and hi - lo < 0.02


def test_fictitious_play_bounds_bracket_value_batched():
    games = np.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 1.0]]])
    _, _, lo, hi = fictitious_play(games, 4000)
    assert np.all(lo <= np.array([0.5, 2 / 3]) + 1e-9)
    assert np.all(hi >= np.array([0.5, 2 / 3]) - 1e-9)
    assert np.all(hi - lo < 0.02)


def test_gae_lambda_one_is_nstep_bootstrapped_return_and_resets_at_done():
    r = np.array([0.0, 1.0, 0.0])
    d = np.array([False, True, False])
    val = np.array([0.5, 0.5, 0.5])
    adv, target = _advantages(r, d, val, last=2.0, gamma=0.9, lam=1.0)
    assert np.isclose(target[1], 1.0)                 # terminal step: no bootstrap
    assert np.isclose(target[0], 0.0 + 0.9 * 1.0)     # does not leak past the done
    assert np.isclose(target[2], 0.0 + 0.9 * 2.0)     # last step bootstraps the critic


def test_a2c_and_ppo_run_and_return_valid_policies():
    g = A10SoccerGame()
    for algo in ("a2c", "ppo"):
        res = train_actor_critic_selfplay(g, algo, iterations=3, rollout_len=20, seed=0)
        pol = res.net0.policy(g.initial_state())
        assert pol.shape == (4,) and np.isclose(pol.sum(), 1.0)
        assert len(res.mean_reward_trace) == 3
