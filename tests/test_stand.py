"""The optional fifth action, STAND (Littman 1994)."""

import numpy as np
import pytest

from soccer_nash.game import Action, SoccerGame
from soccer_nash.nash_q import NashQIteration


def test_n_actions_must_be_four_or_five():
    with pytest.raises(ValueError):
        SoccerGame(n_actions=3)
    with pytest.raises(ValueError):
        SoccerGame(n_actions=6)


def test_action_set_and_joint_actions_track_n_actions():
    assert SoccerGame(n_actions=4).actions() == [
        Action.U, Action.D, Action.L, Action.R
    ]
    g5 = SoccerGame(n_actions=5)
    assert g5.actions()[-1] is Action.STAND
    assert len(g5.joint_actions()) == 25
    assert len(SoccerGame(n_actions=4).joint_actions()) == 16


def test_stand_keeps_a_player_in_place():
    g = SoccerGame(n_actions=5)
    s = (3, 2, 5, 2, 0)
    nxt, reward, done = g.step(s, Action.STAND, Action.STAND)
    assert not done
    assert nxt == (3, 2, 5, 2, 0)
    assert reward == (0, 0)


def test_moving_into_a_standing_carrier_steals_under_random_order():
    # Littman: a move into an occupied cell fails and the ball goes to the
    # stationary player. Carrier stands, defender walks in.
    g = SoccerGame(width=5, height=3, goal_rows=(1,), move_order="random",
                   n_actions=5)
    s = (2, 1, 3, 1, 0)                       # carrier at (2,1), defender at (3,1)
    outs = g.transitions(s, Action.STAND, Action.L)
    # both orderings agree here: defender is blocked, ball flips to the carrier's
    # side stays with the carrier (stationary player keeps it)
    for _p, ns, _r in outs:
        assert ns[0:2] == (2, 1)             # carrier did not move
        assert ns[4] == 0                    # carrier keeps the ball
        assert ns[2:4] == (3, 1)             # defender blocked


def test_carrier_standing_next_to_its_goal_can_score_by_moving_not_standing():
    g = SoccerGame(width=5, height=3, goal_rows=(1,), n_actions=5)
    s = (4, 1, 1, 1, 0)                       # player 0 on the goal column, row 1
    nxt, reward, _ = g.step(s, Action.R, Action.STAND)
    assert nxt == (-1, -1, -1, -1, 0) and reward == (1, -1)
    nxt, reward, done = g.step(s, Action.STAND, Action.STAND)
    assert not done and reward == (0, 0)     # standing never scores


@pytest.fixture(scope="module")
def solved_pair():
    kw = {"width": 5, "height": 4, "goal_rows": (1, 2), "move_order": "random"}
    out = {}
    for n in (4, 5):
        solver = NashQIteration(SoccerGame(n_actions=n, **kw), gamma=0.9,
                                mode="hybrid", tol=1e-10)
        out[n] = (solver, solver.run())
    return out


def test_stand_barely_shifts_where_mixing_happens(solved_pair):
    # Adding STAND mostly changes the *content* of the mix, not its location:
    # the no-pure-saddle sets stay close in size and overlap heavily.
    four = set(solved_pair[4][1].no_saddle_states)
    five = set(solved_pair[5][1].no_saddle_states)
    assert four and five
    assert abs(len(four) - len(five)) <= 0.15 * len(four)
    assert len(four & five) >= 0.6 * len(four | five)


def test_stand_enters_the_mixed_support(solved_pair):
    _solver, result = solved_pair[5]
    used = [
        s for s in result.no_saddle_states
        if result.row_policy[s][4] > 0.02 or result.col_policy[s][4] > 0.02
    ]
    # Littman's Figure 2: the carrier randomizes with "stand" in the support.
    assert len(used) > len(result.no_saddle_states) // 2


def test_five_action_stage_matrix_is_five_by_five(solved_pair):
    solver, result = solved_pair[5]
    m = solver._matrix(next(iter(result.no_saddle_states)), result.values)
    assert m.shape == (5, 5)


def test_policy_svg_draws_a_stand_ring(solved_pair):
    import xml.etree.ElementTree as ET

    solver, result = solved_pair[5]
    game = solver.game
    s = next(
        st for st in result.no_saddle_states
        if result.row_policy[st][4] > 0.1 or result.col_policy[st][4] > 0.1
    )
    from soccer_nash.viz import policy_svg

    svg = policy_svg(game, s, result.row_policy, result.col_policy,
                     result.values[s], title="stand")
    ET.fromstring(svg)
    assert "stroke-dasharray" in svg          # the STAND ring
    assert "hold" in svg


def test_deterministic_five_action_game_still_has_a_pure_saddle():
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), n_actions=5)
    result = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10).run()
    assert result.no_saddle_states == []
    assert np.isclose(result.values[g.initial_state()], 0.0)
