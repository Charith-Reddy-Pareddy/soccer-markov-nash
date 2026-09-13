"""scripts/degeneracy.py -- classifying no-pure-saddle states by whether the
LP-reported support is the whole indifference class or just one vertex of a
larger, degenerate one."""

import importlib.util
import pathlib

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_has_tied_outsider_detects_an_exact_tie():
    d = _load("degeneracy")
    support = frozenset({0, 1})
    values = np.array([1.0, 1.0, 1.0, 0.5])  # action 2 ties 0 and 1 exactly
    assert d._has_tied_outsider(support, values) is True


def test_has_tied_outsider_false_when_strictly_worse():
    d = _load("degeneracy")
    support = frozenset({0, 1})
    values = np.array([1.0, 1.0, 0.9, 0.5])
    assert d._has_tied_outsider(support, values) is False


def test_has_tied_outsider_empty_support_is_false():
    d = _load("degeneracy")
    assert d._has_tied_outsider(frozenset(), np.array([1.0, 2.0])) is False


def test_classify_case5_is_a_carrier_face():
    # state (0, 0, 2, 0, 0) on the canonical board -- documented in
    # positions.md as a state where all four carrier actions tie exactly,
    # even though the LP only weights three of them.
    d = _load("degeneracy")
    from soccer_nash.game import SoccerGame
    from soccer_nash.nash_q import NashQIteration

    game = SoccerGame(**d.CANON)
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run()
    state = (0, 0, 2, 0, 0)
    assert state in result.no_saddle_states

    row = d.classify(solver, state, result.row_policy, result.col_policy,
                      result.values)
    assert row["carrier_degenerate"] is True
    assert "carrier face" in row["kind"]


def test_classify_case9_is_defender_pure_tied():
    # state (0, 2, 1, 2, 0) -- documented in positions.md: the defender's
    # equilibrium is pure R, but D ties it exactly.
    d = _load("degeneracy")
    from soccer_nash.game import SoccerGame
    from soccer_nash.nash_q import NashQIteration

    game = SoccerGame(**d.CANON)
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run()
    state = (0, 2, 1, 2, 0)
    assert state in result.no_saddle_states

    row = d.classify(solver, state, result.row_policy, result.col_policy,
                      result.values)
    assert row["defender_degenerate"] is True
    assert row["defender_support"] == 1
    assert "defender pure-tied" in row["kind"]


def test_classify_case2_is_unique():
    # state (0, 1, 1, 1, 0) -- the "primary example", a genuine matching-
    # pennies core with no reported degeneracy on either side.
    d = _load("degeneracy")
    from soccer_nash.game import SoccerGame
    from soccer_nash.nash_q import NashQIteration

    game = SoccerGame(**d.CANON)
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run()
    state = (0, 1, 1, 1, 0)

    row = d.classify(solver, state, result.row_policy, result.col_policy,
                      result.values)
    assert row["kind"] == "unique"
