import numpy as np

from soccer_nash.exact_play import duality_gap, play, tables
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration


def small_game():
    return SoccerGame(width=4, height=3, move_order="random")


def test_the_exact_equilibrium_has_no_duality_gap_and_a_uniform_pair_does():
    game = small_game()
    states = list(game.states())
    result = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-12).run()
    tab = tables(game, states)
    p0 = np.array([result.row_policy[s] for s in states])
    p1 = np.array([result.col_policy[s] for s in states])
    assert np.abs(duality_gap(p0, p1, tab, 0.9)).max() < 1e-6
    uniform = np.full_like(p0, 0.25)
    assert duality_gap(uniform, uniform, tab, 0.9).max() > 0.1


def test_the_tables_keep_probabilities_and_mark_finished_games():
    game = small_game()
    states = list(game.states())
    nxt, prob, rew = tables(game, states)
    assert np.allclose(prob.sum(-1), 1) and (nxt == len(states)).any()
    assert set(np.unique(rew)) <= {-1.0, 0.0, 1.0}


def test_exact_play_with_a_table_lowers_the_gap():
    rows = play(small_game(), "table", "adam", steps=200, lr=0.05, every=200)
    assert rows[-1][1] < 0.5 * rows[0][1]
