import numpy as np
import pytest

from soccer_nash.matrix_games import solve_zero_sum
from soccer_nash.numerics import (
    certified_value,
    classify_stage_game,
    epsilon_equilibrium,
    essential_subgame,
    is_matching_pennies,
    is_rounding_artifact,
    mixing_entropy,
    rounding_changes_saddle,
    rounding_diagnostic,
    RoundingDiagnostic,
    support_shape,
    value_bracket,
)


def test_mixing_entropy_scale():
    assert mixing_entropy([1.0, 0.0, 0.0]) == 0.0
    assert mixing_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert mixing_entropy([0.25, 0.25, 0.25, 0.25]) == pytest.approx(2.0)
    assert 0.0 < mixing_entropy([0.9, 0.1]) < 0.6      # a near-pure hedge


def test_epsilon_equilibrium_is_zero_at_a_nash():
    _v, p, q = solve_zero_sum(RPS)
    assert epsilon_equilibrium(RPS, p, q) == pytest.approx(0.0, abs=1e-9)
    # a pure strategy in RPS is maximally exploitable
    e = np.eye(3)
    assert epsilon_equilibrium(RPS, e[0], e[0]) == pytest.approx(1.0)


def test_epsilon_equilibrium_matches_the_bracket_half_widths():
    M = np.array([[2.0, -1.0], [0.0, 1.0]])
    p, q = np.array([0.6, 0.4]), np.array([0.7, 0.3])
    b = value_bracket(M, p, q)
    assert epsilon_equilibrium(M, p, q) == pytest.approx(
        max(b.upper - b.mid, b.mid - b.lower)
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


def test_rounding_diagnostic_full_precision_matches_solve_zero_sum():
    diag = rounding_diagnostic(RPS, decimals=(None,))
    value, row, col = solve_zero_sum(RPS)
    assert diag[0].decimals is None
    assert diag[0].pure is False
    assert diag[0].value == pytest.approx(value)
    np.testing.assert_allclose(diag[0].row, row)
    np.testing.assert_allclose(diag[0].col, col)


def test_rounding_diagnostic_reports_every_requested_precision_in_order():
    diag = rounding_diagnostic(RPS, decimals=(None, 2, 1, 0))
    assert [d.decimals for d in diag] == [None, 2, 1, 0]


def test_rounding_diagnostic_reproduces_case3_sensitivity():
    # The concrete Sept-meeting example this function exists to make
    # checkable: Case 3's own matrix stays mixed through 2 decimals but
    # manufactures a pure saddle at 1 decimal, at a value that lands
    # exactly on a round number -- a strong sign the saddle is a rounding
    # artifact, not a real feature of the game.
    case3 = np.array([
        [-0.139279, -0.17079, -0.81, -0.211001],
        [0.173204, -0.185032, -0.499883, -0.18094],
        [-0.099163, -0.156678, -0.14101, -0.211001],
        [-0.109755, -0.17079, -0.81, -0.166529],
    ])
    diag = {d.decimals: d for d in rounding_diagnostic(case3, decimals=(None, 3, 2, 1))}
    assert diag[None].pure is False
    assert diag[3].pure is False
    assert diag[2].pure is False
    assert diag[1].pure is True
    assert diag[1].value == pytest.approx(-0.2, abs=1e-9)
    # 2-decimal rounding barely moves the equilibrium mix...
    np.testing.assert_allclose(diag[2].row, diag[None].row, atol=5e-3)
    np.testing.assert_allclose(diag[2].col, diag[None].col, atol=5e-3)
    # ...while 1-decimal rounding collapses the support from two actions to one.
    assert int((diag[None].row > 1e-6).sum()) == 2
    assert int((diag[1].row > 1e-6).sum()) == 1


def test_is_rounding_artifact_flags_case3s_one_decimal_collapse():
    case3 = np.array([
        [-0.139279, -0.17079, -0.81, -0.211001],
        [0.173204, -0.185032, -0.499883, -0.18094],
        [-0.099163, -0.156678, -0.14101, -0.211001],
        [-0.109755, -0.17079, -0.81, -0.166529],
    ])
    diag = {d.decimals: d for d in rounding_diagnostic(case3, decimals=(None, 2, 1))}
    # 2 decimals barely moves the split within the same two-action support...
    assert is_rounding_artifact(diag[None], diag[2]) is False
    # ...1 decimal collapses pure/mixed classification and the support itself.
    assert is_rounding_artifact(diag[None], diag[1]) is True


def test_is_rounding_artifact_ignores_percentage_drift_within_the_same_support():
    # Same two actions carry weight before and after rounding, just a
    # slightly different split -- not an artifact.
    full = RoundingDiagnostic(None, False, 0.0, np.array([0.635, 0.365, 0.0, 0.0]),
                               np.array([0.5, 0.5, 0.0, 0.0]))
    rounded = RoundingDiagnostic(2, False, 0.0, np.array([0.667, 0.333, 0.0, 0.0]),
                                  np.array([0.5, 0.5, 0.0, 0.0]))
    assert is_rounding_artifact(full, rounded) is False


def test_is_rounding_artifact_flags_a_pure_saddles_action_flip():
    # Case 1: classification stays "pure" at 1 decimal, but the saddle
    # action itself moves from U to L -- still a genuine artifact.
    full = RoundingDiagnostic(None, True, 0.5, np.array([1.0, 0.0, 0.0, 0.0]),
                               np.array([1.0, 0.0, 0.0, 0.0]))
    rounded = RoundingDiagnostic(1, True, 0.5, np.array([0.0, 0.0, 1.0, 0.0]),
                                  np.array([1.0, 0.0, 0.0, 0.0]))
    assert is_rounding_artifact(full, rounded) is True


# -------------------------------------------------------------- subgame shape


def test_essential_subgame_removes_dominated_rows():
    M = np.array([[3.0, 1.0, 4.0], [2.0, 0.0, 1.0], [0.0, -1.0, -2.0]])
    ri, ci, sub = essential_subgame(M)
    assert list(ri) == [0]  # row 0 strictly dominates the others


def test_essential_subgame_removes_dominated_columns():
    # No row dominates; column 1 is strictly worse for the minimiser than
    # column 0 everywhere, so it drops out.
    M = np.array([[1.0, 9.0, 2.0], [8.0, 10.0, 0.0]])
    ri, ci, sub = essential_subgame(M)
    assert list(ri) == [0, 1]
    assert list(ci) == [0, 2]
    assert sub.shape == (2, 2)


def test_support_shape_of_matching_pennies_is_two_by_two():
    assert support_shape(MATCHING_PENNIES) == (2, 2)
    assert support_shape(RPS) == (3, 3)
    assert support_shape(SADDLE) == (1, 1)


def test_is_matching_pennies():
    assert is_matching_pennies(MATCHING_PENNIES)
    assert is_matching_pennies(RPS)
    assert not is_matching_pennies(SADDLE)
    assert not is_matching_pennies(np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]))  # not square
