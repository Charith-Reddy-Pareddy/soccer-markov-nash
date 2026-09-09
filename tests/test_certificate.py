import numpy as np
import pytest

from soccer_nash.certificate import (
    MixedCertificate,
    PureCertificate,
    certify_game,
    classify,
    verify,
)
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration

RPS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
MATCHING_PENNIES = np.array([[1.0, -1.0], [-1.0, 1.0]])
SADDLE = np.array([[4.0, 3.0, 2.0], [1.0, 5.0, 0.0], [3.0, 2.0, 1.0]])


def test_pure_certificate_on_a_textbook_saddle():
    c = certify_game(SADDLE)
    assert isinstance(c, PureCertificate)
    assert c.saddle == (0, 2)
    assert c.value == pytest.approx(2.0)
    assert c.row_slack == pytest.approx(0.0)
    assert c.col_slack == pytest.approx(0.0)
    assert verify(c, SADDLE) == pytest.approx(0.0, abs=1e-12)


def test_mixed_certificate_on_matching_pennies():
    c = certify_game(MATCHING_PENNIES)
    assert isinstance(c, MixedCertificate)
    assert c.gap == pytest.approx(2.0)
    assert c.core is not None
    assert c.core.margin == pytest.approx(2.0)
    assert c.support == (2, 2)
    assert c.entropy == pytest.approx(1.0)
    assert c.best_reply_cycle[0] == c.best_reply_cycle[-1]
    assert verify(c, MATCHING_PENNIES) == pytest.approx(0.0, abs=1e-12)


def test_rps_is_mixed_with_a_length_three_support():
    c = certify_game(RPS)
    assert isinstance(c, MixedCertificate)
    assert c.gap == pytest.approx(2.0)
    assert c.support == (3, 3)
    assert c.iewds_shape == (3, 3)  # nothing is weakly dominated
    assert len(c.best_reply_cycle) >= 4
    assert verify(c, RPS) == pytest.approx(0.0, abs=1e-12)


def test_classify_matches_certify_kind():
    for M in (RPS, MATCHING_PENNIES, SADDLE, np.zeros((3, 3))):
        assert classify(M) == ("mixed" if certify_game(M).kind == "mixed" else "pure")


def test_verify_catches_a_tampered_pure_certificate():
    c = certify_game(SADDLE)
    bad = PureCertificate("pure", (1, 1), c.value, 0.0, 0.0)  # (1,1)=5 is no saddle
    assert verify(bad, SADDLE) > 1.0


def test_certificate_reproduces_the_solver_on_a_small_random_board():
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random")
    r = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10).run()
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    solver.run()
    no_saddle = set(r.no_saddle_states)

    worst = 0.0
    for s in solver._states:
        M = solver._matrix(s, r.values)
        c = certify_game(M)
        assert (c.kind == "mixed") == (s in no_saddle)
        worst = max(worst, verify(c, M))
    assert worst < 1e-6


def test_every_mixed_state_has_a_two_by_two_core_on_the_small_board():
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random")
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    r = solver.run()
    for s in r.no_saddle_states:
        c = certify_game(solver._matrix(s, r.values))
        assert c.core is not None
        # the four crossing inequalities are strict
        A = np.array(c.core.submatrix)
        assert A[0, 0] > A[1, 0] and A[1, 1] > A[0, 1]
        assert A[0, 0] > A[0, 1] and A[1, 1] > A[1, 0]


def test_single_cell_board_is_pure_everywhere():
    g = SoccerGame(width=5, height=3, goal_rows=(1,), move_order="random")
    solver = NashQIteration(g, gamma=0.9, mode="hybrid", tol=1e-10)
    r = solver.run()
    for s in solver._states:
        assert certify_game(solver._matrix(s, r.values)).kind == "pure"
