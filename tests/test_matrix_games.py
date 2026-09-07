import numpy as np
import pytest

from soccer_nash.matrix_games import (
    game_value,
    has_pure_saddle,
    pure_bounds,
    pure_saddle_points,
    security_strategy_row,
    solve_zero_sum,
)

MATCHING_PENNIES = np.array([[1.0, -1.0], [-1.0, 1.0]])
SADDLE_GAME = np.array([[4.0, 3.0, 2.0], [1.0, 5.0, 0.0], [3.0, 2.0, 1.0]])


def test_matching_pennies_has_no_pure_saddle():
    assert not has_pure_saddle(MATCHING_PENNIES)
    assert pure_saddle_points(MATCHING_PENNIES) == []
    lo, hi = pure_bounds(MATCHING_PENNIES)
    assert lo == -1.0 and hi == 1.0


def test_matching_pennies_mixed_value():
    value, p, q = solve_zero_sum(MATCHING_PENNIES)
    assert value == pytest.approx(0.0, abs=1e-7)
    assert p == pytest.approx([0.5, 0.5], abs=1e-6)
    assert q == pytest.approx([0.5, 0.5], abs=1e-6)


def test_saddle_game_detected():
    # Row 0 min is 2 (col 2); col 2 max is 2 (row 0) -> saddle at (0, 2).
    assert has_pure_saddle(SADDLE_GAME)
    assert (0, 2) in pure_saddle_points(SADDLE_GAME)
    lo, hi = pure_bounds(SADDLE_GAME)
    assert lo == hi == 2.0


def test_saddle_game_lp_matches_pure_value():
    value, _, _ = solve_zero_sum(SADDLE_GAME)
    assert value == pytest.approx(2.0, abs=1e-7)


def test_security_strategy_row():
    i, v = security_strategy_row(SADDLE_GAME)
    assert i == 0
    assert v == 2.0


def test_lp_value_is_between_pure_bounds():
    rng = np.random.default_rng(0)
    for _ in range(50):
        A = rng.normal(size=(4, 5))
        lo, hi = pure_bounds(A)
        value, p, q = solve_zero_sum(A)
        assert lo - 1e-6 <= value <= hi + 1e-6
        assert p.sum() == pytest.approx(1.0)
        assert q.sum() == pytest.approx(1.0)
        # Row strategy guarantees at least `value` against any column.
        assert np.min(p @ A) >= value - 1e-6


def test_game_value_matches_solve_zero_sum():
    rng = np.random.default_rng(1)
    for _ in range(30):
        A = rng.normal(size=(4, 4))
        assert game_value(A) == pytest.approx(solve_zero_sum(A)[0], abs=1e-7)


def test_rps_style_value_zero():
    rps = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    value, p, q = solve_zero_sum(rps)
    assert value == pytest.approx(0.0, abs=1e-7)
    assert p == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-6)
