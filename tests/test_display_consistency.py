"""Regression coverage for a real class of bug the group flagged at the
meeting: "some displayed policies look implausible" plus a general note that
the exact solver "needs rounding and consistency fixes for interpretable
mixed strategies". Two concrete, checkable claims follow from that:

1. The LP in `soccer_nash/matrix_games.py` should never leave a spurious
   near-zero probability in a strategy's support -- `_lp_row_value` clips
   negative solver noise to exactly 0 but never clips small positive noise,
   so a `1e-11`-sized numerical artifact could in principle survive and be
   misread as a genuine (if tiny) mixed action.
2. `site/src/explorer/Board.jsx` rounds every displayed action/hold
   percentage independently (`Math.round(p * 100)`); if a state's fully-shown
   support summed to something other than 100% after rounding, the board
   would visibly display probabilities that don't add up.

Both are checked directly against every board `scripts/explorer_data.py`
solves for the live site, not just a hand-picked example -- this file is the
same board-wide checkable-property style as `test_positions_verification.py`.
"""

import numpy as np
import pytest

from scripts.explorer_data import BOARDS
from soccer_nash.game import SoccerGame
from soccer_nash.matrix_games import solve_zero_sum
from soccer_nash.nash_q import NashQIteration

pytestmark = pytest.mark.slow

ACT = ["U", "D", "L", "R"]


def _wall_mask(x: int, y: int, w: int, h: int) -> dict[str, bool]:
    # Mirrors site/src/explorer/helpers.js's wallMask exactly.
    return {"U": y == h - 1, "D": y == 0, "L": x == 0, "R": x == w - 1}


def _rounded_display_total(pol: np.ndarray, wall: dict[str, bool]) -> tuple[int, bool]:
    """Mirror Board.jsx's ActionFan: hold ring + per-action arrows, each
    percentage rounded independently. Returns (total, fully_shown)."""
    hold = sum(pol[i] for i, a in enumerate(ACT) if wall[a])
    parts = []
    if hold >= 0.02:
        parts.append(round(hold * 100))
    fully_shown = hold < 0.02 or hold < 1e-9
    for i, a in enumerate(ACT):
        if wall[a]:
            continue
        p = pol[i]
        if p < 0.02:
            fully_shown = False
            continue
        parts.append(round(p * 100))
    return sum(parts), fully_shown


@pytest.mark.parametrize("board_id", list(BOARDS.keys()))
def test_no_near_zero_lp_noise_in_any_solved_states_support(board_id):
    game = SoccerGame(**BOARDS[board_id])
    solver = NashQIteration(game, gamma=0.9)
    result = solver.run_exact()
    for policies in (result.row_policy, result.col_policy):
        for pol in policies.values():
            noise = [v for v in pol if 0 < v < 1e-6]
            assert not noise, f"board {board_id!r}: near-zero LP noise {noise} in {pol}"


@pytest.mark.parametrize("board_id", list(BOARDS.keys()))
def test_displayed_percentages_sum_to_100_when_fully_shown(board_id):
    kw = BOARDS[board_id]
    game = SoccerGame(**kw)
    solver = NashQIteration(game, gamma=0.9)
    result = solver.run_exact()
    for state, row_pol in result.row_policy.items():
        x0, y0, x1, y1, _b = state
        col_pol = result.col_policy[state]
        for pol, x, y in ((row_pol, x0, y0), (col_pol, x1, y1)):
            wall = _wall_mask(x, y, kw["width"], kw["height"])
            total, fully_shown = _rounded_display_total(pol, wall)
            if fully_shown:
                assert total == 100, (
                    f"board {board_id!r} state {state}: rounded display sums to "
                    f"{total}%, not 100% -- {pol}"
                )


def test_near_tied_matrices_do_not_leave_lp_noise_in_support():
    # Near-ties are where an LP is most likely to return solver noise instead
    # of a clean zero, since the tied face gives it no reason to prefer one
    # vertex -- see docs/numerics.md section 0b.
    rng = np.random.default_rng(0)
    for _ in range(200):
        A = rng.normal(size=(4, 4))
        A[0] = A[1]  # force an exact tie between two rows
        _value, p, q = solve_zero_sum(A)
        for name, arr in (("row", p), ("col", q)):
            noise = [v for v in arr if 0 < v < 1e-9]
            assert not noise, f"{name} strategy has LP noise {noise} for\n{A}"
