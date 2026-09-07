import numpy as np
import pytest

from soccer_nash.matrix_games import solve_zero_sum
from soccer_nash.support_enum import (
    all_equilibria,
    select_equilibrium,
    zero_sum_value,
)

MATCHING_PENNIES = np.array([[1.0, -1.0], [-1.0, 1.0]])
RPS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
BOS_ROW = np.array([[2.0, 0.0], [0.0, 1.0]])
BOS_COL = np.array([[1.0, 0.0], [0.0, 2.0]])


def test_matching_pennies_unique_mixed_equilibrium():
    eqs = all_equilibria(MATCHING_PENNIES, -MATCHING_PENNIES)
    assert len(eqs) == 1
    assert eqs[0].row == pytest.approx([0.5, 0.5])
    assert eqs[0].col == pytest.approx([0.5, 0.5])
    assert eqs[0].row_value == pytest.approx(0.0)


def test_rock_paper_scissors_unique_uniform_equilibrium():
    eqs = all_equilibria(RPS, -RPS)
    assert len(eqs) == 1
    assert eqs[0].row == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_battle_of_the_sexes_has_two_pure_and_one_mixed():
    eqs = all_equilibria(BOS_ROW, BOS_COL)
    assert len(eqs) == 3
    supports = sorted(
        (int((e.row > 1e-6).sum()), int((e.col > 1e-6).sum())) for e in eqs
    )
    assert supports == [(1, 1), (1, 1), (2, 2)]
    # The mixed equilibrium is worse for both players than either pure one.
    mixed = next(e for e in eqs if (e.row > 1e-6).sum() == 2)
    assert mixed.value_sum < 3.0 - 1e-6


def test_largest_sum_selection_avoids_the_bad_mixed_equilibrium():
    eqs = all_equilibria(BOS_ROW, BOS_COL)
    chosen = select_equilibrium(eqs, rule="largest_sum")
    assert chosen.value_sum == pytest.approx(3.0)
    assert int((chosen.row > 1e-6).sum()) == 1  # a pure equilibrium


def test_select_equilibrium_row_rule_and_errors():
    eqs = all_equilibria(BOS_ROW, BOS_COL)
    assert select_equilibrium(eqs, rule="row").row_value == pytest.approx(2.0)
    with pytest.raises(ValueError):
        select_equilibrium(eqs, rule="nonsense")
    with pytest.raises(ValueError):
        select_equilibrium([])


def test_zero_sum_equilibria_all_share_the_game_value():
    rng = np.random.default_rng(1)
    for _ in range(20):
        M = rng.normal(size=(3, 3))
        eqs = all_equilibria(M, -M)
        assert eqs
        v = eqs[0].row_value
        assert all(abs(e.row_value - v) < 1e-6 for e in eqs)


def test_support_enumeration_matches_the_lp_on_zero_sum():
    rng = np.random.default_rng(2)
    for _ in range(30):
        M = rng.normal(size=(4, 4))
        v_se, _, _ = zero_sum_value(M)
        v_lp, _, _ = solve_zero_sum(M)
        assert v_se == pytest.approx(v_lp, abs=1e-7)


def test_pure_coordination_game_finds_both_pure_equilibria():
    A = B = np.array([[1.0, 0.0], [0.0, 1.0]])
    eqs = all_equilibria(A, B)
    pure = sorted(
        tuple(np.argmax(e.row) for _ in [0]) + (int(np.argmax(e.col)),)
        for e in eqs
        if (e.row > 1e-6).sum() == 1
    )
    assert (0, 0) in pure and (1, 1) in pure
