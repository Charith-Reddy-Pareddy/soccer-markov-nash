"""State occupancy under the Nash policy, and the mixed-state concentration."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.occupancy import concentration, visitation
from soccer_nash.viz import occupancy_map_svg


@pytest.fixture(scope="module")
def solved():
    game = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random")
    result = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10).run()
    dist = visitation(game, result.row_policy, result.col_policy, gamma=0.9)
    return game, result, dist


def test_visitation_is_a_distribution(solved):
    _game, _result, dist = solved
    assert dist
    assert all(m > 0 for m in dist.values())
    assert np.isclose(sum(dist.values()), 1.0)


def test_kickoff_carries_real_mass(solved):
    game, _result, dist = solved
    ranked = sorted(dist, key=dist.get, reverse=True)
    assert game.initial_state() in ranked[:5]        # transient but heavily hit


def test_only_reachable_states_get_mass(solved):
    game, _result, dist = solved
    assert len(dist) < len(list(game.states()))     # the path is narrow


def test_mixed_states_are_over_represented_on_the_path(solved):
    _game, result, dist = solved
    mass, uniform = concentration(dist, set(result.no_saddle_states))
    assert mass > uniform                            # concentrated, not diluted
    assert 0.0 < mass <= 1.0


def test_deterministic_game_occupancy_avoids_terminals(solved):
    game = SoccerGame(width=5, height=4, goal_rows=(1, 2))
    result = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10).run()
    dist = visitation(game, result.row_policy, result.col_policy, gamma=0.9)
    assert all(not game.is_terminal(s) for s in dist)


def test_occupancy_map_is_valid_svg(solved):
    game, result, dist = solved
    svg = occupancy_map_svg(game, dist, set(result.no_saddle_states),
                            ball=0, title="t")
    ET.fromstring(svg)
    assert "amber = a mixed state" in svg
