"""move_order="blend": a continuous sweep from deterministic to random."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from soccer_nash.game import Action, SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.viz import P0, line_chart_svg


def test_blend_must_be_a_probability():
    with pytest.raises(ValueError):
        SoccerGame(move_order="blend", blend=1.5)
    with pytest.raises(ValueError):
        SoccerGame(move_order="blend", blend=-0.1)


def test_blend_zero_matches_deterministic():
    det = SoccerGame(move_order="deterministic")
    b0 = SoccerGame(move_order="blend", blend=0.0)
    s = (2, 2, 4, 2, 0)
    for a0 in det.actions():
        for a1 in det.actions():
            assert det.transitions(s, a0, a1) == b0.transitions(s, a0, a1)


def test_blend_one_matches_random():
    rnd = SoccerGame(move_order="random")
    b1 = SoccerGame(move_order="blend", blend=1.0)
    s = (2, 2, 3, 2, 0)
    got = {(ns, r): p for p, ns, r in b1.transitions(s, Action.R, Action.L)}
    want = {(ns, r): p for p, ns, r in rnd.transitions(s, Action.R, Action.L)}
    assert got.keys() == want.keys()
    for k in got:
        assert got[k] == pytest.approx(want[k])


def test_blend_probabilities_sum_to_one():
    g = SoccerGame(move_order="blend", blend=0.3)
    for a0 in g.actions():
        for a1 in g.actions():
            total = sum(p for p, _ns, _r in g.transitions(g.initial_state(), a0, a1))
            assert total == pytest.approx(1.0)


@pytest.mark.slow
def test_a_minority_of_random_order_stays_pure():
    game = SoccerGame(width=5, height=4, goal_rows=(1, 2),
                      move_order="blend", blend=0.4)
    result = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()
    assert result.no_saddle_states == []
    assert np.isclose(result.values[game.initial_state()], 0.0)


@pytest.mark.slow
def test_a_majority_of_random_order_creates_mixing():
    game = SoccerGame(width=5, height=4, goal_rows=(1, 2),
                      move_order="blend", blend=0.75)
    result = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()
    assert len(result.no_saddle_states) > 0


def test_line_chart_is_valid_svg():
    svg = line_chart_svg(
        [("mixed", P0, [(0.0, 0), (0.5, 0), (0.6, 30), (1.0, 20)])],
        x_label="blend", y_label="mixed", title="t", vline=0.5,
    )
    ET.fromstring(svg)
    assert "stroke-dasharray" in svg          # the threshold line
