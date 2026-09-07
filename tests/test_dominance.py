"""Iterated weak-dominance elimination."""

import numpy as np

from soccer_nash.dominance import iewds, is_iewds_solvable

MATCHING_PENNIES = np.array([[1.0, -1.0], [-1.0, 1.0]])
SADDLE = np.array([[4.0, 3.0], [2.0, 1.0]])   # (row 0, col 1) is a pure saddle
DRAW = np.array([                             # a single-cell "draw region" shape
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.3, 0.3, 0.0, 0.0],
])


def test_matching_pennies_is_not_solvable():
    assert not is_iewds_solvable(MATCHING_PENNIES)
    ri, ci, res = iewds(MATCHING_PENNIES)
    assert res.shape == (2, 2)   # nothing eliminated


def test_saddle_game_is_solvable():
    assert is_iewds_solvable(SADDLE)


def test_draw_shape_reduces_to_all_zero_saddle():
    ri, ci, res = iewds(DRAW)
    assert is_iewds_solvable(DRAW)
    assert np.allclose(res, 0.0)          # the positive entry's column is dominated
    assert set(ci.tolist()) <= {2, 3}     # defender keeps the "hold the row" columns


def test_rps_is_not_solvable():
    rps = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    assert not is_iewds_solvable(rps)
