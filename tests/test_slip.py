import pytest

from soccer_nash.certificate import classify
from soccer_nash.game import A10SoccerGame, Action, SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.reachability import reachable_states


def test_slip_validation():
    with pytest.raises(ValueError):
        SoccerGame(slip=-0.1)
    with pytest.raises(ValueError):
        SoccerGame(slip=1.0)
    with pytest.raises(ValueError):
        A10SoccerGame(slip=0.1)  # the exact assignment env forbids variants


def test_slip_zero_is_the_base_game():
    base = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random")
    slip0 = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random",
                       slip=0.0)
    for s in list(base.states())[::17]:
        for a0 in base.actions():
            for a1 in base.actions():
                assert base.transitions(s, a0, a1) == slip0.transitions(s, a0, a1)


def test_slip_transitions_are_a_distribution():
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="deterministic",
                   slip=0.2)
    for s in list(g.states())[::29]:
        for a0 in g.actions():
            for a1 in g.actions():
                outs = g.transitions(s, a0, a1)
                assert sum(p for p, _, _ in outs) == pytest.approx(1.0)
                assert all(p > 0 for p, _, _ in outs)


def test_slip_mixes_in_off_action_outcomes():
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="deterministic",
                   slip=0.25)
    s = (1, 1, 3, 2, 0)
    chosen = {ns for _p, ns, _r in
              SoccerGame(width=5, height=4, goal_rows=(1, 2)).transitions(
                  s, Action.U, Action.U)}
    slipped = {ns for _p, ns, _r in g.transitions(s, Action.U, Action.U)}
    assert slipped > chosen  # strictly more reachable next states


@pytest.mark.slow
def test_slip_creates_mixing_under_deterministic_order():
    """The goal-width switch is move-order specific: action-independent noise
    produces mixed stage games even with deterministic resolution."""
    def mixed(slip):
        g = SoccerGame(width=5, height=5, goal_rows=(2,), move_order="deterministic",
                       slip=slip)
        solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
        r = solver.run()
        reach = reachable_states(g)
        return sum(classify(solver._matrix(x, r.values)) == "mixed" for x in reach)

    assert mixed(0.0) == 0          # one goal cell, deterministic -> pure
    assert mixed(0.15) > 0          # slip breaks it
