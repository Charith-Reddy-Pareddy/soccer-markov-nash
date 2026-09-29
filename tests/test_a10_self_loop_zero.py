"""Regression test for a question raised about the A10-deterministic board's
Q matrices: several states (514 of 2380, the most common shape on the board)
have every one of their 16 Q-matrix cells exactly 0.0. That is not a rounding
artifact -- it is an algebraic consequence of Littman's own reward convention
(goals worth +-1, nothing else, gamma=0.9; Littman, "Markov games as a
framework for multi-agent reinforcement learning," ICML 1994) whenever the
equilibrium joint action is a genuine self-loop: the Bellman equation
V = 0 + gamma*V has the unique solution V = 0 for any gamma < 1. This test
checks that fact directly against the exact solve, for every state on the
board, not just the one worked by hand in docs/a10_cases.html.
"""

from soccer_nash.game import MOVE_ACTIONS, SoccerGame
from soccer_nash.nash_q import NashQIteration

BOARD = {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "deterministic"}


def test_self_loop_states_have_exactly_zero_value():
    game = SoccerGame(**BOARD)
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run_exact()

    checked = 0
    for state in game.states():
        a0 = MOVE_ACTIONS[int(result.row_policy[state].argmax())]
        a1 = MOVE_ACTIONS[int(result.col_policy[state].argmax())]
        outs = game.transitions(state, a0, a1)
        assert len(outs) == 1  # deterministic board: always a single outcome
        _prob, next_state, reward = outs[0]
        if next_state == state and reward == (0, 0):
            assert result.values[state] == 0.0, (
                f"{state}: a genuine zero-reward self-loop must have V == 0 "
                f"exactly, got {result.values[state]!r}"
            )
            checked += 1

    # Sanity: the property was actually exercised, not vacuously true.
    assert checked > 0


def test_all_zero_q_matrix_count_matches_documented_finding():
    """docs/a10_cases.html's own case 8 ("the dead zone") cites 514 of 2380
    states as having a fully flat, all-zero Q matrix -- the single most
    common shape on the board. Pinned here so a future change to the game
    rules or solver that silently shifts this count gets caught."""
    game = SoccerGame(**BOARD)
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run_exact()

    all_zero = 0
    for state in game.states():
        M = solver._matrix(state, result.values)
        if (M == 0.0).all():
            all_zero += 1

    assert all_zero == 514
