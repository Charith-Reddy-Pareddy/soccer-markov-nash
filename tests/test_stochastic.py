import numpy as np
import pytest

from soccer_nash.game import Action, SoccerGame


@pytest.fixture
def game():
    return SoccerGame(move_order="random")


def test_rejects_unknown_move_order():
    with pytest.raises(ValueError):
        SoccerGame(move_order="sideways")


def test_transition_probabilities_sum_to_one(game):
    for s in game.states():
        for a0 in Action:
            for a1 in Action:
                total = sum(p for p, _, _ in game.transitions(s, a0, a1))
                assert total == pytest.approx(1.0)


def test_contested_square_is_a_coin_flip(game):
    # Carrier and defender both step into the cell between them.
    s = (2, 2, 4, 2, 0)
    outcomes = {(ns, r): p for p, ns, r in game.transitions(s, Action.R, Action.L)}
    assert outcomes == {
        ((3, 2, 4, 2, 0), (0, 0)): pytest.approx(0.5),  # carrier wins the cell
        ((2, 2, 3, 2, 1), (0, 0)): pytest.approx(0.5),  # defender wins it + ball
    }


def test_swap_attempt_freezes_positions_and_splits_ball(game):
    s = (3, 2, 4, 2, 0)
    outcomes = {(ns, r): p for p, ns, r in game.transitions(s, Action.R, Action.L)}
    assert outcomes == {
        ((3, 2, 4, 2, 0), (0, 0)): pytest.approx(0.5),
        ((3, 2, 4, 2, 1), (0, 0)): pytest.approx(0.5),
    }


def test_open_goal_score_is_certain(game):
    s = (6, 2, 0, 0, 0)
    outcomes = game.transitions(s, Action.R, Action.U)
    assert outcomes == [(1.0, (-1, -1, -1, -1, 0), (1, -1))]


def test_non_carrier_cannot_score(game):
    s = (6, 2, 0, 2, 1)  # player 0 at right edge, no ball
    for _, ns, _ in game.transitions(s, Action.R, Action.L):
        assert not game.is_terminal(ns) or ns[4] == 1


def test_step_sampling_is_reproducible(game):
    s = (2, 2, 4, 2, 0)
    a = [
        SoccerGame(move_order="random").step(
            s, Action.R, Action.L, rng=np.random.default_rng(7)
        )
        for _ in range(2)
    ]
    assert a[0] == a[1]


def test_step_sampling_hits_both_outcomes(game):
    s = (2, 2, 4, 2, 0)
    rng = np.random.default_rng(0)
    seen = {game.step(s, Action.R, Action.L, rng=rng)[0] for _ in range(50)}
    assert (3, 2, 4, 2, 0) in seen and (2, 2, 3, 2, 1) in seen


def test_chaser_follows_into_vacated_cell(game):
    # Carrier steps up out of (2, 2); the chaser aims at the cell it vacated
    # and should be allowed in (collision is checked against live positions).
    s = (2, 2, 3, 2, 1)
    outcomes = {(ns, r): p for p, ns, r in game.transitions(s, Action.U, Action.L)}
    # first = carrier: carrier -> (2, 3), then chaser -> (2, 2).
    assert ((2, 3, 2, 2, 1), (0, 0)) in outcomes


def test_every_transition_yields_a_legal_state(game):
    legal = set(game.states())
    for s in game.states():
        for a0 in Action:
            for a1 in Action:
                for _, ns, _ in game.transitions(s, a0, a1):
                    if game.is_terminal(ns):
                        continue
                    assert ns in legal, f"{s} {a0} {a1} -> {ns}"
                    assert (ns[0], ns[1]) != (ns[2], ns[3])
