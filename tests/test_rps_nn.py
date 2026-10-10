import numpy as np
import pytest
import torch

from soccer_nash import rps_nn


def test_the_uniform_policy_is_unexploitable_and_a_pure_one_is_exploited_for_two():
    u = torch.full((3,), 1 / 3)
    assert rps_nn.exploitability(u, u) == pytest.approx(0.0, abs=1e-6)
    rock = torch.tensor([1.0, 0.0, 0.0])
    assert rps_nn.exploitability(rock, rock) == pytest.approx(2.0)


def test_every_mode_returns_a_series_per_iteration_and_rejects_unknown_modes():
    for mode in rps_nn.MODES:
        r = rps_nn.run(mode, iterations=6, batch=8, snap_every=2)
        assert len(r["current"]) == len(r["aggregate"]) == len(r["exploitability"]) == 6
        assert all(0.0 <= v <= 1.0 for v in r["aggregate"])
    with pytest.raises(ValueError):
        rps_nn.run("argmax", iterations=1)


def test_the_pure_aggregate_is_the_frequency_of_each_snapshots_top_move():
    nets = [rps_nn.make_net() for _ in range(4)]
    agg = rps_nn.aggregate(nets, pure=True)
    tops = [int(rps_nn.probs(n).argmax()) for n in nets]
    assert np.allclose(agg.numpy(), np.bincount(tops, minlength=3) / 4)
    soft = rps_nn.aggregate(nets, pure=False)
    assert soft.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_standard_training_reports_the_current_network_as_the_aggregate():
    r = rps_nn.run("standard", iterations=5, batch=8)
    assert r["current"] == r["aggregate"]
