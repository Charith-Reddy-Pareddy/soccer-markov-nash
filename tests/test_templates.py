"""Geometric templates for the mixed-strategy states."""

import numpy as np
import pytest

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.symmetry import mirror_state
from soccer_nash.templates import (
    describe_action,
    equilibrium_support,
    matching_pennies_pattern,
    mirror_reduce,
    mixed_state_templates,
)

MATCHING_PENNIES = np.array([[1.0, -1.0], [-1.0, 1.0]])
SADDLE = np.array([[4.0, 3.0], [2.0, 1.0]])  # (0,1) is a pure saddle
RPS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])


def test_matching_pennies_pattern_crosses():
    mp = matching_pennies_pattern(MATCHING_PENNIES)
    assert mp.is_matching_pennies
    assert mp.p_row == pytest.approx(0.5)
    assert mp.q_col == pytest.approx(0.5)
    assert mp.value == pytest.approx(0.0)


def test_saddle_pattern_does_not_cross():
    mp = matching_pennies_pattern(SADDLE)
    assert not mp.is_matching_pennies
    assert not (mp.row_crosses and mp.col_crosses)


def test_equilibrium_support_sizes():
    ri, ci, sub = equilibrium_support(MATCHING_PENNIES)
    assert (len(ri), len(ci)) == (2, 2)
    assert equilibrium_support(RPS)[2].shape == (3, 3)
    sri, sci, _ = equilibrium_support(SADDLE)
    assert (len(sri), len(sci)) == (1, 1)


def test_mirror_reduce_halves_a_mirror_closed_set():
    game = SoccerGame(width=5, height=5)
    states = list(game.states())[:40]
    closed = set(states) | {mirror_state(s, game.width) for s in states}
    reps = mirror_reduce(game, sorted(closed))
    assert len(reps) == len(closed) // 2
    # no representative is the mirror of another
    rep_set = set(reps)
    assert not any(mirror_state(r, game.width) in rep_set for r in reps)


def test_describe_action_reads_the_carrier_frame():
    # player 0 carries, attacks +x; player 1 sits directly ahead, same row.
    s = (2, 2, 4, 2, 0)
    assert describe_action(SoccerGame(), s, 0, 3) == "advance"   # R
    assert describe_action(SoccerGame(), s, 0, 2) == "retreat"   # L
    assert describe_action(SoccerGame(), s, 1, 2) == "block"     # L -> (3,2) threat cell


def test_describe_action_is_mirror_invariant():
    game = SoccerGame()
    s = (1, 2, 4, 2, 0)
    m = mirror_state(s, game.width)
    # action 3 (R) for player 0 mirrors to action 2 (L) for player 1's role;
    # the gloss should be the same because it is carrier-frame.
    assert describe_action(game, s, 0, 3) == describe_action(game, m, 1, 2)


@pytest.mark.slow
def test_templates_of_the_random_game_are_matching_pennies():
    game = SoccerGame(move_order="random")
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run()
    mixed = list(result.no_saddle_states)

    templates = mixed_state_templates(
        game, mixed, lambda st: solver._matrix(st, result.values)
    )
    assert sum(t.count for t in templates) == len(mixed) == 94

    two_by_two = [t for t in templates if t.mp is not None]
    assert two_by_two, "expected some 2x2-support templates"
    # every 2x2-support template is a genuine matching-pennies subgame
    assert all(t.mp.is_matching_pennies for t in two_by_two)
    assert sum(t.count for t in two_by_two) == 68
