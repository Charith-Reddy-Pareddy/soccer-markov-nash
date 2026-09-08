"""The 'rate' reward objective: goal reset, expected goal difference."""

import numpy as np
import pytest

from soccer_nash.game import A10SoccerGame, Action, SoccerGame
from soccer_nash.nash_q import NashQIteration


def test_scoring_must_be_win_or_rate():
    with pytest.raises(ValueError):
        SoccerGame(scoring="draws")


def test_a10_game_rejects_the_rate_objective():
    with pytest.raises(ValueError):
        A10SoccerGame(scoring="rate")


def test_win_mode_goal_is_terminal():
    g = SoccerGame(width=5, height=3, goal_rows=(1,), scoring="win")
    s = (4, 1, 1, 1, 0)                       # player 0 on the goal column
    ns, reward, done = g.step(s, Action.R, Action.U)
    assert done and g.is_terminal(ns) and reward == (1, -1)


def test_rate_mode_goal_resets_with_the_ball_to_the_conceding_team():
    g = SoccerGame(width=5, height=3, goal_rows=(1,), scoring="rate")
    s = (4, 1, 1, 1, 0)                       # player 0 about to score
    ns, reward, done = g.step(s, Action.R, Action.U)
    assert not done and not g.is_terminal(ns)
    assert reward == (1, -1)                  # the goal still pays
    # restart: both at kickoff cells, ball with player 1 (the team that conceded)
    mid = g.height // 2
    assert ns == (0, mid, g.width - 1, mid, 1)


def test_rate_mode_transitions_never_terminate():
    g = SoccerGame(width=5, height=3, goal_rows=(1,), scoring="rate")
    for s in g.states():
        for a0, a1 in g.joint_actions():
            for _p, ns, _r in g.transitions(s, a0, a1):
                assert not g.is_terminal(ns)


@pytest.fixture(scope="module")
def solved():
    board = {"width": 5, "height": 4, "goal_rows": (1, 2), "move_order": "random"}
    out = {}
    for scoring in ("win", "rate"):
        solver = NashQIteration(SoccerGame(scoring=scoring, **board), gamma=0.9,
                                mode="hybrid", tol=1e-10)
        out[scoring] = solver.run()
    return out


def test_mixed_region_is_identical_under_win_and_rate(solved):
    assert set(solved["win"].no_saddle_states) == set(solved["rate"].no_saddle_states)
    assert solved["win"].no_saddle_states


def test_rate_compresses_the_value_range(solved):
    wv = np.fromiter(solved["win"].values.values(), dtype=float)
    rv = np.fromiter(solved["rate"].values.values(), dtype=float)
    assert rv.max() - rv.min() < wv.max() - wv.min()
    assert rv.min() > wv.min()               # no state is as bad as a loss


def test_deterministic_rate_game_still_has_a_pure_saddle():
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), scoring="rate")
    result = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10).run()
    assert result.no_saddle_states == []
