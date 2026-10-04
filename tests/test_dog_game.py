import torch

from soccer_nash import dog_game as dg


def test_policy_actions_are_in_range():
    pol = dg.PolarPolicy(dg.DOG_SPEED)
    x = dg.features(torch.rand(32, 2), torch.rand(32, 2), 3)
    theta, frac = pol.act(x)
    assert theta.shape == (32,) and ((frac > 0) & (frac < 1)).all()
    assert torch.isfinite(pol.log_prob(x, theta, frac)).all()


def test_move_stays_in_the_square_and_respects_the_radius():
    pos = torch.tensor([[0.99, 0.01]])
    new = dg.move(pos, torch.tensor([0.0]), torch.tensor([0.06]))
    assert (new >= 0).all() and (new <= 1).all()
    mid = dg.move(torch.tensor([[0.5, 0.5]]), torch.tensor([1.0]), torch.tensor([0.04]))
    assert abs(float(torch.linalg.norm(mid - 0.5)) - 0.04) < 1e-6


def test_straight_line_dog_catches_a_random_sheep_and_random_play_does_not():
    assert dg.play(dg.greedy_dog_act, dg.random_act, n_games=50)["capture_rate"] == 1.0
    assert dg.play(dg.random_act, dg.random_act, n_games=50)["capture_rate"] < 0.1


def test_ppo_selfplay_runs():
    dog, sheep = dg.train_ppo_selfplay(iterations=1, n_envs=4)
    x = dg.features(torch.rand(2, 2), torch.rand(2, 2), 0)
    assert dog.act(x)[0].shape == (2,) and sheep.act(x, greedy=True)[1].shape == (2,)
