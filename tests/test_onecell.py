"""The single-goal-cell pure-saddle certificate."""

import pytest

from soccer_nash.game import Action, SoccerGame
from soccer_nash.onecell import certify, guard_action


def test_guard_action_requires_a_single_goal_row():
    with pytest.raises(ValueError):
        guard_action(SoccerGame(goal_rows=(1, 2, 3)), (0, 2, 6, 2, 0))


def test_guard_heads_for_the_goal_row_then_the_goal_cell():
    game = SoccerGame(width=7, height=5, goal_rows=(2,))
    # defender (player 1) below the goal row -> climb toward it
    assert guard_action(game, (0, 2, 3, 0, 0)) == Action.U
    # defender on the goal row, left of the goal cell T=(6,2) -> slide right
    assert guard_action(game, (0, 2, 3, 2, 0)) == Action.R


def test_certificate_small_boards():
    for w, h in [(3, 3), (5, 3), (3, 5)]:
        c = certify(w, h, gamma=0.9)
        assert c.guard_slack < 1e-6
        assert c.iewds_failures == 0
        assert c.ok
        assert c.kickoff_value == pytest.approx(0.0)


@pytest.mark.slow
def test_certificate_holds_on_larger_boards_and_discounts():
    for w, h in [(7, 5), (5, 7), (9, 5)]:
        for gamma in (0.5, 0.9, 0.99):
            c = certify(w, h, gamma)
            assert c.ok, (w, h, gamma, c.guard_slack, c.iewds_failures)
