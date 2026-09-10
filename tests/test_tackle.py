import pytest

from soccer_nash.certificate import classify
from soccer_nash.game import A10SoccerGame, Action, SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.reachability import reachable_states


def test_tackle_prob_validation():
    with pytest.raises(ValueError):
        SoccerGame(move_order="tackle", tackle_prob=-0.1)
    with pytest.raises(ValueError):
        SoccerGame(move_order="tackle", tackle_prob=1.5)
    with pytest.raises(ValueError):
        A10SoccerGame(move_order="tackle")


def test_no_challenge_resolves_deterministically():
    tk = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="tackle")
    det = SoccerGame(width=5, height=4, goal_rows=(1, 2))
    # a state where the defender is nowhere near the carrier
    s = (0, 0, 4, 3, 0)
    for a0 in tk.actions():
        for a1 in tk.actions():
            assert tk.transitions(s, a0, a1) == det.transitions(s, a0, a1)


def test_a_challenge_is_a_two_outcome_duel():
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="tackle",
                   tackle_prob=0.4)
    # carrier at (2,2) with ball, defender at (1,2) moving Right onto it
    s = (2, 2, 1, 2, 0)
    outs = g.transitions(s, Action.U, Action.R)  # defender challenges
    probs = sorted(round(p, 6) for p, _, _ in outs)
    assert probs == [0.4, 0.6]
    # one outcome gives the defender the ball, the other leaves it with player 0
    ball = {ns[4] for _p, ns, _r in outs}
    assert ball == {0, 1}


def test_challenge_success_shoves_the_carrier_back():
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="tackle",
                   tackle_prob=1.0)  # always wins
    s = (2, 2, 1, 2, 0)
    outs = g.transitions(s, Action.STAND if g.n_actions == 5 else Action.U, Action.R)
    assert len(outs) == 1
    _p, ns, _r = outs[0]
    assert ns[4] == 1                      # defender has the ball
    assert (ns[0], ns[1]) == (3, 2)        # carrier shoved from x=2 to x=3
    assert (ns[2], ns[3]) == (2, 2)        # defender took the vacated cell


def test_transitions_are_always_a_distribution():
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="tackle",
                   tackle_prob=0.5)
    for s in list(g.states())[::31]:
        for a0 in g.actions():
            for a1 in g.actions():
                outs = g.transitions(s, a0, a1)
                assert sum(p for p, _, _ in outs) == pytest.approx(1.0)


@pytest.mark.slow
def test_tackle_creates_mixing_at_any_goal_width():
    def mixed(goal_rows):
        g = SoccerGame(width=5, height=4, goal_rows=goal_rows,
                       move_order="tackle", tackle_prob=0.5)
        solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
        r = solver.run()
        reach = reachable_states(g)
        return sum(classify(solver._matrix(x, r.values)) == "mixed" for x in reach)

    assert mixed((2,)) > 0       # single goal cell -- still mixed under tackle
    assert mixed((1, 2)) > 0
