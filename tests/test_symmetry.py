import numpy as np
import pytest

from soccer_nash.game import Action, SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.symmetry import (
    canonical_pairs,
    flip_action,
    flip_distribution,
    mirror_state,
)


@pytest.fixture
def game():
    return SoccerGame()


def test_mirror_is_an_involution(game):
    for s in list(game.states())[::37]:
        assert mirror_state(mirror_state(s, game.width), game.width) == s


def test_no_state_is_its_own_mirror(game):
    for s in game.states():
        assert mirror_state(s, game.width) != s  # b always flips


def test_canonical_pairs_cover_everything_once(game):
    reps, image_of = canonical_pairs(game)
    assert len(reps) * 2 == len(list(game.states()))
    for s in game.states():
        assert image_of[s] in reps
        assert image_of[mirror_state(s, game.width)] == image_of[s]


def test_flip_action_swaps_left_and_right():
    assert flip_action(Action.L) == Action.R
    assert flip_action(Action.R) == Action.L
    assert flip_action(Action.U) == Action.U
    assert np.array_equal(flip_distribution([0.1, 0.2, 0.3, 0.4]), [0.1, 0.2, 0.4, 0.3])


def test_transition_is_mirror_equivariant(game):
    for s in list(game.states())[::23]:
        for a0 in Action:
            for a1 in Action:
                ns, _, _ = game.step(s, a0, a1)
                ms, _, _ = game.step(
                    mirror_state(s, game.width), flip_action(a1), flip_action(a0)
                )
                assert ms == mirror_state(ns, game.width)


def test_run_symmetric_matches_run(game):
    full = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10).run()
    sym = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10).run_symmetric()
    assert max(abs(full.values[s] - sym.values[s]) for s in full.values) < 1e-12
    assert set(full.no_saddle_states) == set(sym.no_saddle_states)
    for s in list(full.values)[::53]:
        assert np.array_equal(full.row_policy[s], sym.row_policy[s])


def test_run_symmetric_rejects_stochastic_move_order():
    with pytest.raises(ValueError):
        NashQIteration(SoccerGame(move_order="random")).run_symmetric()
