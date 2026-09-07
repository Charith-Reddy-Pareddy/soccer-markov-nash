"""State reachability from the kickoff."""

from soccer_nash.game import SoccerGame
from soccer_nash.reachability import reachable_states


def test_kickoff_is_reachable_and_terminals_excluded():
    game = SoccerGame(width=5, height=5)
    reach = reachable_states(game)
    assert game.initial_state() in reach
    assert all(not game.is_terminal(s) for s in reach)


def test_every_valid_state_is_reachable_on_the_standard_boards():
    # Both players move freely each turn, so the whole legal state space is
    # reachable -- which is why the phase diagram's mixed fraction over
    # reachable states equals the fraction over all states.
    for mo in ("deterministic", "random"):
        game = SoccerGame(width=5, height=5, move_order=mo)
        assert len(reachable_states(game)) == sum(1 for _ in game.states())


def test_reachable_from_a_custom_start():
    game = SoccerGame(width=5, height=5)
    reach = reachable_states(game, start=(0, 0, 4, 4, 0))
    assert (0, 0, 4, 4, 0) in reach
    assert len(reach) > 1
