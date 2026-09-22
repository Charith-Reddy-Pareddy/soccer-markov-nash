"""Independent numeric verification of the exact claims docs/positions.md and
docs/positions.pdf make -- both "every printed policy is a genuine Nash
equilibrium" (not just a plausible-looking fractional split) and "the
fourteen documented cases are classified pure/mixed as the doc claims". This
is the canonical board every one of them (except 8, 11, 12, which use their
own board) is drawn from, so it verifies the actual content on the page, not
a smaller stand-in board.
"""

import numpy as np
import pytest

from soccer_nash.game import SoccerGame
from soccer_nash.matrix_games import solve_zero_sum
from soccer_nash.nash_q import NashQIteration
from soccer_nash.numerics import epsilon_equilibrium

# A full hybrid solve of the 2380-state canonical board takes tens of
# seconds -- the same class of cost the `slow` marker exists for elsewhere
# in this suite (`make test-all`, not the fast `make test`).
pytestmark = pytest.mark.slow

CANON = {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "random"}


@pytest.fixture(scope="module")
def solved():
    g = SoccerGame(**CANON)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run_exact()
    return solver, result

# state -> should this be a no-pure-saddle (mixed) state? -- matches the
# `should_be_mixed` assertions in scripts/positions.py's own case list, and
# the labels in docs/positions.md.
DOCUMENTED_CASES = {
    (4, 0, 5, 0, 0): False,   # Case 1 -- pure, for contrast
    (0, 1, 1, 1, 0): True,    # Case 2 -- the typical mix
    (1, 1, 1, 0, 1): True,    # Case 3 -- L/R indifference
    (0, 0, 1, 1, 0): True,    # Case 4 -- the corner duel
    (0, 0, 2, 0, 0): True,    # Case 5 -- 3-action mix
    (1, 1, 2, 0, 1): True,    # Case 6 -- near-pure hedge
    (0, 2, 1, 2, 0): True,    # Case 9 -- asymmetric mix
    (0, 2, 2, 2, 1): True,    # Case 10 -- three-lane mix
    (1, 1, 3, 1, 1): True,    # Case 13 -- template 3
    (0, 2, 2, 2, 0): True,    # Case 14 -- template 8, rarest shape
}


def test_documented_cases_are_classified_as_positions_md_claims(solved):
    _solver, result = solved
    mixed = set(result.no_saddle_states)
    for state, should_be_mixed in DOCUMENTED_CASES.items():
        assert (state in mixed) == should_be_mixed, (
            f"{state}: docs/positions.md and the solver disagree on pure/mixed"
        )


def test_every_printed_policy_on_the_canonical_board_is_a_genuine_equilibrium(solved):
    """For every no-pure-saddle state on the exact board positions.md uses,
    the printed (row_policy, col_policy) must satisfy the Nash indifference
    condition to solver precision -- not just look like a clean fraction."""
    solver, result = solved
    no_saddle = result.no_saddle_states
    assert len(no_saddle) > 0  # otherwise this test isn't exercising anything

    worst = 0.0
    for s in no_saddle:
        M = solver._matrix(s, result.values)
        eps = epsilon_equilibrium(M, result.row_policy[s], result.col_policy[s])
        worst = max(worst, eps)
    assert worst < 1e-6


def test_case2_and_case3_indifference_matches_positions_pdf_exactly(solved):
    """Cross-check against the specific numbers printed in docs/positions.pdf
    for Case 2 and Case 3, so a future solver change that silently shifts
    these numbers is caught here, not just noticed by a reader."""
    _solver, result = solved

    case2 = (0, 1, 1, 1, 0)
    p, q = result.row_policy[case2], result.col_policy[case2]
    np.testing.assert_allclose(p, [0.635, 0.365, 0.0, 0.0], atol=5e-4)
    np.testing.assert_allclose(q, [0.365, 0.0, 0.0, 0.635], atol=5e-4)

    case3 = (1, 1, 1, 0, 1)
    p, q = result.row_policy[case3], result.col_policy[case3]
    np.testing.assert_allclose(p, [0.0, 0.18, 0.82, 0.0], atol=5e-4)
    np.testing.assert_allclose(q, [0.0, 0.0, 0.077, 0.923], atol=5e-4)


def test_case13_and_case14_indifference_matches_positions_pdf_exactly(solved):
    """Same cross-check as Case 2/3, for the two templates (3 and 8) added to
    fill in the last gaps in docs/templates.md's 8-template coverage."""
    _solver, result = solved

    case13 = (1, 1, 3, 1, 1)
    p, q = result.row_policy[case13], result.col_policy[case13]
    np.testing.assert_allclose(p, [0.656, 0.0, 0.0, 0.344], atol=5e-4)
    np.testing.assert_allclose(q, [0.823, 0.0, 0.177, 0.0], atol=5e-4)

    case14 = (0, 2, 2, 2, 0)
    p, q = result.row_policy[case14], result.col_policy[case14]
    np.testing.assert_allclose(p, [0.675, 0.325, 0.0, 0.0], atol=5e-4)
    np.testing.assert_allclose(q, [0.0, 0.0, 1.0, 0.0], atol=5e-4)


def test_case3_joint_support_reduction_reproduces_the_full_equilibrium(solved):
    """docs/positions.md's Case 3 "reading the matrix" table reduces the
    4x4 game to carrier{L, R} x defender{D, L} -- each player's own
    equilibrium support, not an arbitrary pair of actions. That reduction
    is only valid if the 2x2 game, solved entirely on its own, reproduces
    the same equilibrium as the full game -- checked directly here, not
    just asserted in prose. (An earlier version of that table used
    carrier{L, R} x defender{L, R} -- mixing one of the defender's real
    actions, L, with one it never plays, R, while omitting D. Solving that
    version does not reproduce the real equilibrium; solved below to show
    exactly why the corrected reduction was necessary.)"""
    solver, result = solved
    state = (1, 1, 1, 0, 1)
    M = solver._matrix(state, result.values)  # rows=player0(defender), cols=player1(carrier)
    acts = ["U", "D", "L", "R"]

    def idx(a: str) -> int:
        return acts.index(a)

    # carrier{L, R} x defender{D, L}: rows=carrier, cols=defender, so
    # transpose-and-negate M (player0's payoff) into carrier's payoff.
    correct = np.array([
        [-M[idx("D"), idx("L")], -M[idx("L"), idx("L")]],
        [-M[idx("D"), idx("R")], -M[idx("L"), idx("R")]],
    ])
    _value, carrier_p, defender_p = solve_zero_sum(correct)
    np.testing.assert_allclose(carrier_p, [0.077, 0.923], atol=5e-3)
    np.testing.assert_allclose(defender_p, [0.18, 0.82], atol=5e-3)

    # The earlier, invalid carrier{L, R} x defender{L, R} version does NOT
    # reproduce the real equilibrium -- confirms the correction was real.
    invalid = np.array([
        [-M[idx("L"), idx("L")], -M[idx("R"), idx("L")]],
        [-M[idx("L"), idx("R")], -M[idx("R"), idx("R")]],
    ])
    _value, bad_carrier_p, bad_defender_p = solve_zero_sum(invalid)
    assert not np.allclose(bad_carrier_p, [0.077, 0.923], atol=5e-3)
