import numpy as np
import pytest

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration


@pytest.fixture(scope="module")
def hybrid_result():
    game = SoccerGame()
    return game, NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()


def test_converges(hybrid_result):
    _, r = hybrid_result
    assert r.iterations < 100


def test_values_bounded(hybrid_result):
    _, r = hybrid_result
    v = np.array(list(r.values.values()))
    assert v.min() >= -1.0 - 1e-9
    assert v.max() <= 1.0 + 1e-9


def test_value_decays_with_distance_to_goal(hybrid_result):
    game, r = hybrid_result
    # Carrier on the goal row, defender parked in the far corner.
    assert r.values[(6, 2, 0, 0, 0)] == pytest.approx(1.0)
    assert r.values[(5, 2, 0, 0, 0)] == pytest.approx(0.9)
    assert r.values[(4, 2, 0, 0, 0)] == pytest.approx(0.81)


def test_defender_between_carrier_and_goal_neutralizes(hybrid_result):
    _, r = hybrid_result
    assert r.values[(1, 2, 5, 2, 0)] == pytest.approx(0.0)


def test_pure_equilibrium_exists_in_deterministic_game(hybrid_result):
    _, r = hybrid_result
    assert r.pure_equilibrium_exists
    assert r.saddle_fraction == pytest.approx(1.0)


def test_role_swap_antisymmetry(hybrid_result):
    game, r = hybrid_result
    w = game.width - 1
    for (x0, y0, x1, y1, b), v in list(r.values.items())[::37]:
        mirror = (w - x1, y1, w - x0, y0, 1 - b)
        assert r.values[mirror] == pytest.approx(-v, abs=1e-6)


SMALL = SoccerGame(width=5, height=3, goal_rows=(1,), max_steps=100)


def test_modes_agree_when_saddles_exist():
    hybrid = NashQIteration(SMALL, gamma=0.9, mode="hybrid", tol=1e-9).run()
    mixed = NashQIteration(SMALL, gamma=0.9, mode="mixed", tol=1e-9).run()
    worst = max(abs(hybrid.values[s] - mixed.values[s]) for s in hybrid.values)
    assert worst < 1e-6


def test_pure_mode_is_lower_bound():
    hybrid = NashQIteration(SMALL, gamma=0.9, mode="hybrid", tol=1e-9).run()
    pure = NashQIteration(SMALL, gamma=0.9, mode="pure", tol=1e-9).run()
    for s in hybrid.values:
        assert pure.values[s] <= hybrid.values[s] + 1e-6


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        NashQIteration(SoccerGame(), mode="bogus")
