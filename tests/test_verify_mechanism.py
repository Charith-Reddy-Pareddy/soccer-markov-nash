"""Fast slices of scripts/verify_mechanism.py -- the full run is `make verify`."""

import numpy as np
import pytest

from soccer_nash.certificate import classify
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.reachability import reachable_states


def _mixed_count(game, gamma=0.9):
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-10)
    r = solver.run()
    reach = reachable_states(game)
    return sum(classify(solver._matrix(s, r.values)) == "mixed" for s in reach)


def test_single_cell_zero_mixed_wider_positive():
    assert _mixed_count(SoccerGame(width=5, height=3, goal_rows=(1,),
                                   move_order="random")) == 0
    assert _mixed_count(SoccerGame(width=5, height=4, goal_rows=(1, 2),
                                   move_order="random")) > 0


@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.995])
def test_switch_invariant_to_gamma(gamma):
    assert _mixed_count(SoccerGame(width=5, height=3, goal_rows=(1,),
                                   move_order="random"), gamma) == 0
    assert _mixed_count(SoccerGame(width=5, height=4, goal_rows=(1, 2),
                                   move_order="random"), gamma) > 0


@pytest.mark.parametrize("scoring", ["win", "rate"])
def test_switch_invariant_to_reward_objective(scoring):
    assert _mixed_count(SoccerGame(width=5, height=3, goal_rows=(1,),
                                   move_order="random", scoring=scoring)) == 0
    assert _mixed_count(SoccerGame(width=5, height=4, goal_rows=(1, 2),
                                   move_order="random", scoring=scoring)) > 0


def test_switch_invariant_to_the_stand_action():
    assert _mixed_count(SoccerGame(width=5, height=3, goal_rows=(1,),
                                   move_order="random", n_actions=5)) == 0
    assert _mixed_count(SoccerGame(width=5, height=4, goal_rows=(1, 2),
                                   move_order="random", n_actions=5)) > 0


def test_perturbation_does_not_flip_genuine_mixed_states():
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random")
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    r = solver.run()
    rng = np.random.default_rng(0)
    for s in reachable_states(g):
        M = solver._matrix(s, r.values)
        gap = M.max(axis=0).min() - M.min(axis=1).max()
        if gap > 1e-4:  # a genuinely mixed stage game
            noisy = classify(M + rng.uniform(-1e-6, 1e-6, size=M.shape))
            assert noisy == "mixed"


@pytest.mark.slow
def test_full_verification_script_passes(tmp_path, monkeypatch):
    import scripts.verify_mechanism as vm

    monkeypatch.setattr(vm, "OUT", tmp_path / "cert.json")
    monkeypatch.setattr(vm, "FIG", tmp_path / "mechanism.svg")
    monkeypatch.setattr(vm, "SINGLE_CELL", [(5, 3)])
    monkeypatch.setattr(vm, "WIDER", [(5, 4, (1, 2))])
    single = vm.part_single_cell(0.9)
    wider, sample = vm.part_wider(0.9)
    perturb = vm.part_perturbation()
    assert single["ok"] and wider["ok"] and perturb["ok"]
    assert sample is not None
