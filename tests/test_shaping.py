import numpy as np
import pytest

from soccer_nash.best_response import BestResponse
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.opponents import part2_opponent
from soccer_nash.shaping import PotentialShaping, StepPossessionBonus
from soccer_nash.simulate import play_deterministic


@pytest.fixture(scope="module")
def game():
    return SoccerGame()


# ------------------------------------------------------------------- potential


def test_phi_is_zero_on_terminal_states(game):
    sh = PotentialShaping()
    assert sh.phi(game, (-1, -1, -1, -1, 0)) == 0.0
    assert sh.phi(game, (-1, -1, -1, -1, 1)) == 0.0


def test_reward_delta_does_not_discount_the_current_potential(game):
    sh = PotentialShaping(0.1, 0.1)
    s, s_next = (3, 2, 4, 2, 0), (4, 2, 4, 3, 0)
    gamma = 0.9
    expected = gamma * sh.phi(game, s_next) - sh.phi(game, s)
    assert sh.reward_delta(game, s, s_next, gamma) == pytest.approx(expected)
    # The wrong form gamma*(phi(s') - phi(s)) would give a different number here.
    wrong = gamma * (sh.phi(game, s_next) - sh.phi(game, s))
    assert not sh.reward_delta(game, s, s_next, gamma) == pytest.approx(wrong)


def test_phi_sign_follows_possession(game):
    sh = PotentialShaping(w_ball=0.1, w_advance=0.0)
    assert sh.phi(game, (3, 2, 4, 2, 0)) > 0  # player 0 has the ball
    assert sh.phi(game, (3, 2, 4, 2, 1)) < 0  # player 1 has the ball


def test_potential_shaping_shifts_best_response_value_by_minus_phi(game):
    sh = PotentialShaping(0.1, 0.1)
    base = BestResponse(game, part2_opponent, me=0, gamma=0.9, tol=1e-12).solve()
    shaped = BestResponse(
        game, part2_opponent, me=0, gamma=0.9, tol=1e-12, shaping=sh
    ).solve()
    err = max(
        abs(shaped.values[s] - (base.values[s] - sh.phi(game, s)))
        for s in base.values
    )
    assert err < 1e-9


def test_potential_shaping_keeps_best_response_optimal(game):
    sh = PotentialShaping(0.1, 0.1)
    base = BestResponse(game, part2_opponent, me=0, gamma=0.9, tol=1e-12).solve()
    shaped = BestResponse(
        game, part2_opponent, me=0, gamma=0.9, tol=1e-12, shaping=sh
    ).solve()

    def base_q(s, a):
        a_opp = part2_opponent(game, s, 1)
        ns, (r0, _), _ = game.step(s, a, a_opp)
        return r0 + (0.0 if game.is_terminal(ns) else 0.9 * base.values[ns])

    for s in base.values:
        best = max(base_q(s, a) for a in range(4))
        assert base_q(s, shaped.policy[s]) >= best - 1e-7

    rollout = play_deterministic(game, shaped.policy, part2_opponent, me=0)
    assert rollout.winner == 0


@pytest.mark.slow
def test_potential_shaping_preserves_nash_value_and_saddle_structure(game):
    sh = PotentialShaping(0.1, 0.1)
    base = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10).run()
    shaped = NashQIteration(
        game, gamma=0.9, mode="hybrid", tol=1e-10, shaping=sh
    ).run()

    err = max(
        abs(shaped.values[s] - (base.values[s] - sh.phi(game, s)))
        for s in base.values
    )
    assert err < 1e-7
    assert set(shaped.no_saddle_states) == set(base.no_saddle_states)


# --------------------------------------------------------------- non-potential


def test_step_bonus_changes_the_solution():
    small = SoccerGame(width=5, height=3, goal_rows=(1,))
    base = NashQIteration(small, gamma=0.9, mode="hybrid", tol=1e-8).run()
    bonus = NashQIteration(
        small, gamma=0.9, mode="hybrid", tol=1e-8, shaping=StepPossessionBonus(0.05)
    ).run()
    s0 = small.initial_state()
    # The draw at kickoff becomes a positive value: the carrier is now paid to
    # hold the ball rather than having to score.
    assert base.values[s0] == pytest.approx(0.0)
    assert bonus.values[s0] > 0.3


def test_step_bonus_is_zero_sum(game):
    sh = StepPossessionBonus(0.05)
    # Whatever it adds for player 0 holding the ball it subtracts for player 1.
    assert sh.reward_delta(game, (3, 2, 4, 2, 0), (4, 2, 3, 2, 0), 0.9) == pytest.approx(0.05)
    assert sh.reward_delta(game, (3, 2, 4, 2, 1), (4, 2, 3, 2, 1), 0.9) == pytest.approx(-0.05)
