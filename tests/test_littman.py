"""scripts/littman.py and scripts/mixing.py -- the Littman reproduction and the
value-of-mixing study."""

import importlib.util
import pathlib

import numpy as np
import pytest

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


littman = _load("littman")


@pytest.fixture(scope="module")
def solved5():
    game, solver, result = littman.solve(5)
    return game, solver, result


def test_figure_2_state_is_an_adjacent_carrier_mix(solved5):
    game, _solver, result = solved5
    s = littman.figure_2_state(game, result)
    x0, y0, x1, y1, b = s
    (cx, cy), (dx, dy) = ((x0, y0), (x1, y1)) if b == 0 else ((x1, y1), (x0, y0))
    assert abs(cx - dx) + abs(cy - dy) == 1                 # defender adjacent
    carrier_pol = result.row_policy[s] if b == 0 else result.col_policy[s]
    assert carrier_pol[4] > 0.1                             # stand in the support
    assert (carrier_pol > 0.02).sum() == 2                  # a two-action mix


@pytest.mark.slow
def test_stand_lifts_the_kickoff_value():
    lo = littman.solve(4)[2]
    hi = littman.solve(5)[2]
    k = SoccerGame(**littman.BOARD).initial_state()
    assert hi.values[k] > lo.values[k]


@pytest.mark.slow
def test_value_of_mixing_is_positive_and_bounded():
    game = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random")
    hy = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10).run()
    pu = NashQIteration(game, gamma=0.9, mode="pure", tol=1e-10).run()
    vom = np.array([hy.values[s] - pu.values[s] for s in hy.no_saddle_states])
    assert (vom > -1e-9).all()          # pure play never beats the Nash value
    assert vom.mean() > 0.0             # and generally loses ground
    assert vom.max() < 1.0
