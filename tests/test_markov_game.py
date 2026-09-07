import numpy as np
import pytest

from soccer_nash.markov_game import solve_markov_game


def test_repeated_battle_of_the_sexes():
    A = np.array([[2.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, 2.0]])

    def transition(s, a0, a1):
        return [(1.0, "x")]

    def reward(s, a0, a1, s_next):
        return float(A[a0, a1]), float(B[a0, a1])

    r = solve_markov_game(["x"], (2, 2), transition, reward, gamma=0.9)
    # A pure-coordination equilibrium, discounted forever: 2 / (1 - 0.9) = 20.
    assert r.row_values["x"] + r.col_values["x"] == pytest.approx(30.0)
    assert "x" in r.multi_equilibrium_states


def test_coordination_markov_game_reaches_the_goal():
    def transition(s, a0, a1):
        if s == 0 and a0 == 0 and a1 == 0:
            return [(1.0, "done")]
        return [(1.0, 0)]

    def reward(s, a0, a1, s_next):
        return (1.0, 1.0) if s_next == "done" else (0.0, 0.0)

    r = solve_markov_game([0], (2, 2), transition, reward, gamma=0.9)
    assert r.row_values[0] == pytest.approx(1.0)
    assert np.argmax(r.row_policy[0]) == 0 and np.argmax(r.col_policy[0]) == 0


@pytest.mark.slow
def test_zero_sum_markov_game_matches_the_lp_solver():
    from soccer_nash.game import SoccerGame
    from soccer_nash.nash_q import NashQIteration

    game = SoccerGame(width=5, height=3, goal_rows=(1,))
    nq = NashQIteration(game, gamma=0.9, mode="mixed", tol=1e-9).run()

    states = list(game.states())

    def transition(s, a0, a1):
        return [(p, ns) for p, ns, _ in game.transitions(s, a0, a1)]

    def reward(s, a0, a1, s_next):
        for p, ns, (r0, r1) in game.transitions(s, a0, a1):
            if ns == s_next:
                return r0, r1
        return 0.0, 0.0

    r = solve_markov_game(states, (4, 4), transition, reward, gamma=0.9, tol=1e-9)
    worst = max(abs(r.row_values[s] - nq.values[s]) for s in states)
    assert worst < 1e-6
