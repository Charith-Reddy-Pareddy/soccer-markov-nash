import numpy as np
import pytest

from soccer_nash.matrix_games import solve_zero_sum
from soccer_nash.numerics import (
    certified_value,
    classify_stage_game,
    essential_subgame,
    is_matching_pennies,
    rounding_changes_saddle,
    support_shape,
    value_bracket,
)

MATCHING_PENNIES = np.array([[1.0, -1.0], [-1.0, 1.0]])
RPS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
SADDLE = np.array([[4.0, 3.0, 2.0], [1.0, 5.0, 0.0], [3.0, 2.0, 1.0]])


# ------------------------------------------------------------- value bracket


def test_bracket_brackets_the_true_value():
    rng = np.random.default_rng(0)
    for _ in range(40):
        M = rng.normal(size=(4, 4))
        v, p, q = solve_zero_sum(M)
        b = value_bracket(M, p, q)
        assert b.lower <= v + 1e-7
        assert v <= b.upper + 1e-7
        assert b.gap >= -1e-9


def test_bracket_collapses_at_equilibrium():
    v, p, q = solve_zero_sum(MATCHING_PENNIES)
    b = value_bracket(MATCHING_PENNIES, p, q)
    assert b.gap == pytest.approx(0.0, abs=1e-7)
    assert b.lower == pytest.approx(b.upper, abs=1e-7)


def test_certified_value_error_is_a_real_bound():
    for M in (RPS, SADDLE, MATCHING_PENNIES):
        v, err = certified_value(M)
        true_v = solve_zero_sum(M)[0]
        assert abs(v - true_v) <= err + 1e-9


def test_mid_uses_the_row_player_layout():
    # Asymmetric pure-saddle game: value is M[0, 2] = 2, not the transpose's 3.
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 1.0])
    assert value_bracket(SADDLE, p, q).mid == pytest.approx(SADDLE[0, 2])


def test_bracket_gap_is_positive_for_a_bad_strategy():
    # Row player wrongly commits to a pure sub-optimal action in matching pennies.
    b = value_bracket(MATCHING_PENNIES, np.array([1.0, 0.0]), np.array([0.5, 0.5]))
    assert b.lower == pytest.approx(-1.0)  # opponent can punish the pure commit
    assert b.gap == pytest.approx(1.0)  # certified 1.0 away from equilibrium


# --------------------------------------------------------------- classification


def test_classification_of_textbook_games():
    assert classify_stage_game(MATCHING_PENNIES) == "mixed"
    assert classify_stage_game(RPS) == "mixed"
    assert classify_stage_game(SADDLE) == "pure"
    assert classify_stage_game(np.zeros((3, 3))) == "degenerate"


def test_classification_tolerance_is_scale_aware():
    tiny_rps = RPS * 1e-4  # same structure, deep in a gamma^k band
    # A relative tolerance treats it like RPS...
    assert classify_stage_game(tiny_rps, rel_tol=1e-6) == "mixed"
    # ...an absolute check of 1e-3 would call it degenerate; a scaled one does not.
    assert classify_stage_game(tiny_rps, rel_tol=1e-2) == "degenerate"


def test_rounding_flips_saddle_on_a_scaled_rps():
    scaled = np.array([[0.01, -0.02, 0.03], [0.02, 0.01, -0.02], [-0.03, 0.02, 0.01]])
    assert classify_stage_game(scaled) == "mixed"
    assert rounding_changes_saddle(scaled, decimals=1) is True
    assert rounding_changes_saddle(RPS, decimals=1) is False


# -------------------------------------------------------------- subgame shape


def test_essential_subgame_removes_dominated_rows():
    M = np.array([[3.0, 1.0, 4.0], [2.0, 0.0, 1.0], [0.0, -1.0, -2.0]])
    ri, ci, sub = essential_subgame(M)
    assert list(ri) == [0]  # row 0 strictly dominates the others


def test_support_shape_of_matching_pennies_is_two_by_two():
    assert support_shape(MATCHING_PENNIES) == (2, 2)
    assert support_shape(RPS) == (3, 3)
    assert support_shape(SADDLE) == (1, 1)


def test_is_matching_pennies():
    assert is_matching_pennies(MATCHING_PENNIES)
    assert is_matching_pennies(RPS)
    assert not is_matching_pennies(SADDLE)
    assert not is_matching_pennies(np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]))  # not square
