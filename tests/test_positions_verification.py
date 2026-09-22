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

from soccer_nash.game import Action, SoccerGame
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


def test_case3_two_q_entries_expand_exactly_to_their_bellman_sum(solved):
    """The two worked-out entries docs/positions.md shows expanded as
    sum(P * (r + gamma * V(next))) -- Q(U,U), off either player's support,
    and Q(D,U), in the defender's actual 18% support and notable because one
    of its two successors is the state itself -- reproduce the matrix cell
    to the same precision the matrix is printed at, using the real
    game.transitions() outcomes and the solver's own V, not a hand re-derivation
    that could silently drift from what run_exact() actually computed."""
    solver, result = solved
    game = solver.game
    state = (1, 1, 1, 0, 1)
    gamma = solver.gamma
    M = solver._matrix(state, result.values)

    def bellman_sum(a0, a1):
        total = 0.0
        for prob, ns, reward in game.transitions(state, a0, a1):
            v = 0.0 if game.is_terminal(ns) else result.values[ns]
            total += prob * (reward[0] + gamma * v)
        return total

    assert bellman_sum(Action.U, Action.U) == pytest.approx(M[0, 0], abs=1e-6)
    assert bellman_sum(Action.D, Action.U) == pytest.approx(M[1, 0], abs=1e-6)
    # Q(D,U)'s defining feature: one of its two successors is the state itself.
    outcomes = game.transitions(state, Action.D, Action.U)
    assert any(ns == state for _prob, ns, _reward in outcomes)


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
    """A standalone mathematical check, not tied to any currently-displayed
    table: restricting Case 3's 4x4 game to carrier{L, R} x defender{D, L}
    (each player's own equilibrium support, not an arbitrary pair of
    actions) and solving that 2x2 game entirely on its own reproduces the
    same equilibrium as the full game. An earlier version of
    docs/positions.md displayed exactly this reduction as the primary
    explanation of Case 3; per the project's Sept 2026 research meeting
    feedback ("write down the whole game" -- a reduced game invites a
    circular-looking argument, solve the full 4x4 first and explain the
    equilibrium from there), that display was removed from the main text.
    The reduction itself was never wrong, only its use as the headline
    explanation -- kept here as a regression check on the underlying fact,
    since it is still true and still useful to know. (Also shows that an
    earlier, genuinely invalid version of the same idea -- carrier{L, R} x
    defender{L, R}, mixing one of the defender's real actions, L, with one
    it never plays, R, while omitting D -- does NOT reproduce the real
    equilibrium, which is exactly why that particular reduction was wrong.)"""
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


# Every 4x4 Q matrix printed in docs/positions.md/.html/.pdf, for all 14
# documented cases -- checked against a fresh solve of the exact board each
# one is drawn from, at the full 6-decimal precision the page prints them
# to. This is the "rounding trick" / "these 4x4 matrices are still
# incorrect" concern raised in the Sept 2026 meeting, checked directly
# rather than assumed fixed: every case matched on inspection (no rounding
# or staleness bug was found), and this test exists so that stays true.
_ALL_CASE_MATRICES = {
    1: {
        "state": (4, 0, 5, 0, 0),
        "M": [
            [0.234446, 0.316945, 0.522973, 0.271536],
            [0.150878, 0.211001, 0.211001, 0.222362],
            [0.161034, 0.182213, 0.171623, 0.189901],
            [0.012406, -0.095109, 0.057946, 0.097586],
        ],
    },
    2: {
        "state": (0, 1, 1, 1, 0),
        "M": [
            [0.085599, 0.478297, 0.28567, 0.099198],
            [0.10926, 0.07633, 0.085331, 0.085599],
            [0.103381, 0.103381, 0.084811, 0.083738],
            [0.08248, 0.187381, -0.108448, -0.070696],
        ],
    },
    3: {
        "state": (1, 1, 1, 0, 1),
        "M": [
            [-0.139279, -0.17079, -0.81, -0.211001],
            [0.173204, -0.185032, -0.499883, -0.18094],
            [-0.099163, -0.156678, -0.14101, -0.211001],
            [-0.109755, -0.17079, -0.81, -0.166529],
        ],
    },
    4: {
        "state": (0, 0, 1, 1, 0),
        "M": [
            [0.103381, 0.103381, -0.407594, 0.083738],
            [0.10926, 0.07633, 0.10085, 0.085599],
            [0.10926, 0.07633, 0.10085, 0.085599],
            [0.112055, -0.075068, 0.112055, 0.088459],
        ],
    },
    5: {
        "state": (0, 0, 2, 0, 0),
        "M": [
            [0.083738, 0.095109, 0.103381, 0.11022],
            [0.085599, 0.076363, 0.07633, 0.085599],
            [0.085599, 0.076363, 0.07633, 0.085599],
            [0.088459, 0.095109, -0.088213, 0.11022],
        ],
    },
    6: {
        "state": (1, 1, 2, 0, 1),
        "M": [
            [-0.316945, -0.211001, -0.17079, -0.182213],
            [-0.316945, -0.211001, -0.044962, -0.182213],
            [-0.247069, -0.211001, -0.156678, -0.182213],
            [0.178022, -0.166529, -0.17079, -0.135158],
        ],
    },
    7: {
        "state": (5, 1, 6, 1, 1),
        "M": [
            [-0.085599, -0.10926, -0.08248, -0.103381],
            [-0.478297, -0.07633, -0.187381, -0.103381],
            [-0.099198, -0.085599, 0.070696, -0.083738],
            [-0.28567, -0.085331, 0.108448, -0.084811],
        ],
    },
    8: {
        "state": (2, 3, 3, 3, 1),
        "M": [
            [0.011944, -0.07235, 0.018582, 0.007062],
            [-0.005242, 0.022044, 0.080389, 0.025575],
            [-0.015879, -0.008641, -0.018582, 0.006356],
            [0.041869, -0.000279, 0.041869, 0.039427],
        ],
    },
    9: {
        "state": (0, 2, 1, 2, 0),
        "M": [
            [0.084811, 0.478297, 0.290839, 0.095109],
            [0.478297, 0.084811, 0.290839, 0.095109],
            [0.093043, 0.093043, 0.085599, 0.093043],
            [0.082303, 0.082303, -0.122276, -0.071591],
        ],
    },
    10: {
        "state": (0, 2, 2, 2, 1),
        "M": [
            [-0.247069, -0.330151, -0.366481, -0.228062],
            [-0.330151, -0.247069, -0.366481, -0.228062],
            [-0.271536, -0.271536, -0.330151, -0.219944],
            [-0.316945, -0.316945, -0.109966, -0.205131],
        ],
    },
    11: {
        "state": (4, 4, 5, 4, 0),
        "M": [
            [0.116139, 0.104525, 0.169846, 0.135972],
            [0.094073, 0.415483, 0.8145, 0.45],
            [0.094073, 0.104525, 0.0, 0.104525],
            [0.169846, 0.5, -0.670721, 0.5],
        ],
    },
    12: {
        "state": (1, 3, 1, 4, 1),
        "M": [
            [0.273301, 0.03143, -0.396398, -0.039469],
            [-0.00596, -0.068847, -0.083433, -0.020201],
            [-0.001041, -0.056711, -0.039473, -0.0153],
            [-0.069053, -0.449923, -0.449665, -0.091802],
        ],
    },
    13: {
        "state": (1, 1, 3, 1, 1),
        "M": [
            [-0.205131, -0.182213, -0.316945, -0.182213],
            [-0.283411, -0.182213, -0.316945, -0.182213],
            [-0.228062, -0.182213, -0.247069, -0.182213],
            [-0.262831, -0.135158, -0.049186, -0.162618],
        ],
    },
    14: {
        "state": (0, 2, 2, 2, 0),
        "M": [
            [0.083738, 0.144198, 0.103381, 0.112892],
            [0.144198, 0.083738, 0.103381, 0.112892],
            [0.099198, 0.099198, 0.085599, 0.099198],
            [0.155749, 0.155749, -0.109966, 0.132828],
        ],
    },
}


def _check_case(solver, result, case):
    """rows = player 0, cols = player 1, player 0's payoff -- fixed for
    every state, never reoriented by who has the ball, matching
    `docs/positions.md`'s own printed convention (and `scripts/positions.py`'s
    `_raw_matrix`, not duplicated here since no transform is needed: this
    IS what `solver._matrix` already returns)."""
    state, expected_M = case["state"], np.array(case["M"])
    actual_M = solver._matrix(state, result.values)
    np.testing.assert_allclose(actual_M, expected_M, atol=5e-6)


def test_canonical_board_case_matrices_match_positions_md_exactly(solved):
    """Cases 1-7, 9, 10, 13, 14 -- every canonical-board case's printed 4x4
    Q matrix, checked at once against a fresh solve."""
    solver, result = solved
    for n in (1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14):
        _check_case(solver, result, _ALL_CASE_MATRICES[n])


def test_tackle_board_case8_matrix_matches_positions_md_exactly():
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="tackle", tackle_prob=0.5)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run_exact()
    _check_case(solver, result, _ALL_CASE_MATRICES[8])


def test_territory_board_case11_matrix_matches_positions_md_exactly():
    g = SoccerGame(width=7, height=5, goal_rows=(1, 2, 3), move_order="deterministic",
                   scoring="territory", territory_reward=0.05)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run_exact()
    _check_case(solver, result, _ALL_CASE_MATRICES[11])


def test_slip_board_case12_matrix_matches_positions_md_exactly():
    g = SoccerGame(width=5, height=5, goal_rows=(2,), move_order="deterministic", slip=0.15)
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run_exact()
    _check_case(solver, result, _ALL_CASE_MATRICES[12])
