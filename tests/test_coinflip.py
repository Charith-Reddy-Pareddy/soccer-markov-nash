import pytest

from soccer_nash.game import Action, SoccerGame
from soccer_nash.nash_q import NashQIteration


@pytest.fixture
def game():
    return SoccerGame(move_order="coinflip")


def test_move_order_accepted():
    assert SoccerGame(move_order="coinflip").move_order == "coinflip"
    with pytest.raises(ValueError):
        SoccerGame(move_order="tossup")


def test_transition_probabilities_sum_to_one(game):
    for s in game.states():
        for a0 in Action:
            for a1 in Action:
                total = sum(p for p, _, _ in game.transitions(s, a0, a1))
                assert total == pytest.approx(1.0)


def test_contested_cell_is_a_coin_flip(game):
    s = (2, 2, 4, 2, 0)
    outcomes = {(ns, r): p for p, ns, r in game.transitions(s, Action.R, Action.L)}
    assert outcomes == {
        ((3, 2, 4, 2, 1), (0, 0)): pytest.approx(0.5),  # player 0 takes the cell
        ((2, 2, 3, 2, 0), (0, 0)): pytest.approx(0.5),  # player 1 takes it
    }


def test_swap_always_happens_ball_is_flipped(game):
    s = (3, 2, 4, 2, 0)
    outcomes = {(ns, r): p for p, ns, r in game.transitions(s, Action.R, Action.L)}
    assert outcomes == {
        ((4, 2, 3, 2, 1), (0, 0)): pytest.approx(0.5),
        ((4, 2, 3, 2, 0), (0, 0)): pytest.approx(0.5),
    }


def test_uncontested_move_is_deterministic(game):
    s = (0, 2, 6, 2, 0)
    assert game.transitions(s, Action.U, Action.U) == [(1.0, (0, 3, 6, 3, 0), (0, 0))]


def test_every_transition_yields_a_legal_state(game):
    legal = set(game.states())
    for s in game.states():
        for a0 in Action:
            for a1 in Action:
                for _, ns, _ in game.transitions(s, a0, a1):
                    if game.is_terminal(ns):
                        continue
                    assert ns in legal
                    assert (ns[0], ns[1]) != (ns[2], ns[3])


def test_deterministic_is_coinflip_with_the_carrier_winning():
    g = SoccerGame()
    for s in list(g.states())[::53]:
        x0, y0, x1, y1, b = s
        for a0 in Action:
            for a1 in Action:
                det = g._resolve_deterministic((x0, y0), (x1, y1), a0, a1, b)
                won = g._resolve_with_winner((x0, y0), (x1, y1), a0, a1, b, winner=b)
                assert det == won


def test_coinflip_matches_the_deterministic_equilibrium():
    det = NashQIteration(
        SoccerGame(move_order="deterministic"), gamma=0.9, mode="hybrid", tol=1e-10
    ).run()
    coin = NashQIteration(
        SoccerGame(move_order="coinflip"), gamma=0.9, mode="hybrid", tol=1e-10
    ).run()
    assert coin.pure_equilibrium_exists
    assert max(abs(det.values[s] - coin.values[s]) for s in det.values) < 1e-9
