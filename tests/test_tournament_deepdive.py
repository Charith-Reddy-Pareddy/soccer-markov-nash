"""scripts/tournament_deepdive.py -- the patched-policy causal test."""

import importlib.util
import pathlib

import numpy as np
import pytest

from soccer_nash.exploit import onehot_policy

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_patched_policy_uses_nash_only_at_the_mixed_states():
    t = _load("tournament_deepdive")
    states = [(0, 0, 1, 1, 0), (1, 1, 1, 0, 1), (2, 2, 2, 2, 0)]
    nash = {s: np.array([0.0, 0.6, 0.4, 0.0]) for s in states}
    greedy = onehot_policy(dict.fromkeys(states, 0), n_actions=4)
    mixed = {states[0], states[2]}

    patched = t.patched_policy(nash, greedy, mixed)

    np.testing.assert_array_equal(patched[states[0]], nash[states[0]])
    np.testing.assert_array_equal(patched[states[2]], nash[states[2]])
    np.testing.assert_array_equal(patched[states[1]], greedy[states[1]])


@pytest.mark.slow
def test_deepdive_reproduces_the_headline_robustness_split():
    t = _load("tournament_deepdive")
    from soccer_nash.game import SoccerGame

    game = SoccerGame(scoring="rate", **t.BOARD)
    nash, no_saddle, greedy = t.build(game, t.GAMMA)
    minimax_v, _ = t.challenger_value(game, nash.row_policy, t.GAMMA)
    greedy_v, _ = t.challenger_value(game, greedy, t.GAMMA)
    patched = t.patched_policy(nash.row_policy, greedy, no_saddle)
    patched_v, _ = t.challenger_value(game, patched, t.GAMMA)

    assert minimax_v > 0.0          # minimax is never exploitable
    assert greedy_v < 0.0           # greedy collapses to its challenger
    # patching only the mixed states should recover *some* of the gap,
    # without necessarily reaching all the way to minimax's own robustness
    assert greedy_v < patched_v <= minimax_v
    assert 0 < len(no_saddle) < len(list(game.states()))
