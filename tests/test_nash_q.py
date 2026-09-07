import numpy as np
import pytest

from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration


@pytest.fixture(scope="module")
def hybrid_result():
    game = SoccerGame()
    return game, NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()


def test_converges(hybrid_result):
    _, r = hybrid_result
    assert r.iterations < 100


def test_values_bounded(hybrid_result):
    _, r = hybrid_result
    v = np.array(list(r.values.values()))
    assert v.min() >= -1.0 - 1e-9
    assert v.max() <= 1.0 + 1e-9


def test_value_decays_with_distance_to_goal(hybrid_result):
    game, r = hybrid_result
    # Carrier on the goal row, defender parked in the far corner.
    assert r.values[(6, 2, 0, 0, 0)] == pytest.approx(1.0)
    assert r.values[(5, 2, 0, 0, 0)] == pytest.approx(0.9)
    assert r.values[(4, 2, 0, 0, 0)] == pytest.approx(0.81)


def test_immediate_scoring_reward_is_undiscounted():
    # A state with a forced score next move is worth exactly +1, not gamma.
    game = SoccerGame()
    for gamma in (0.5, 0.9, 0.99):
        r = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-9).run()
        assert r.values[(6, 2, 0, 0, 0)] == pytest.approx(1.0)
        assert r.values[(5, 2, 0, 0, 0)] == pytest.approx(gamma)


def test_defender_between_carrier_and_goal_neutralizes(hybrid_result):
    _, r = hybrid_result
    assert r.values[(1, 2, 5, 2, 0)] == pytest.approx(0.0)


def test_pure_equilibrium_exists_in_deterministic_game(hybrid_result):
    _, r = hybrid_result
    assert r.pure_equilibrium_exists
    assert r.saddle_fraction == pytest.approx(1.0)


def test_role_swap_antisymmetry(hybrid_result):
    game, r = hybrid_result
    w = game.width - 1
    for (x0, y0, x1, y1, b), v in list(r.values.items())[::37]:
        mirror = (w - x1, y1, w - x0, y0, 1 - b)
        assert r.values[mirror] == pytest.approx(-v, abs=1e-6)


SMALL = SoccerGame(width=5, height=3, goal_rows=(1,), max_steps=100)


def test_modes_agree_when_saddles_exist():
    hybrid = NashQIteration(SMALL, gamma=0.9, mode="hybrid", tol=1e-9).run()
    mixed = NashQIteration(SMALL, gamma=0.9, mode="mixed", tol=1e-9).run()
    worst = max(abs(hybrid.values[s] - mixed.values[s]) for s in hybrid.values)
    assert worst < 1e-6


def test_pure_mode_is_lower_bound():
    hybrid = NashQIteration(SMALL, gamma=0.9, mode="hybrid", tol=1e-9).run()
    pure = NashQIteration(SMALL, gamma=0.9, mode="pure", tol=1e-9).run()
    for s in hybrid.values:
        assert pure.values[s] <= hybrid.values[s] + 1e-6


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        NashQIteration(SoccerGame(), mode="bogus")


# ------------------------------------------------------------ policy iteration


def test_pure_strategies_pick_the_saddle_point():
    # Saddle at (row 0, col 2): row mins are [2, 0, 1, 2] (maximin 2 at row 0),
    # col maxes are [5, 5, 2, 7] (minimax 2 at col 2).
    m = np.array(
        [
            [4.0, 3.0, 2.0, 3.0],
            [1.0, 5.0, 0.0, 6.0],
            [3.0, 2.0, 1.0, 4.0],
            [5.0, 4.0, 2.0, 7.0],
        ]
    )
    p, q = NashQIteration._pure_strategies(m)
    assert np.argmax(p) == 0  # would be 3 if the row axis were flipped
    assert np.argmax(q) == 2


def test_deterministic_hybrid_needs_no_linear_program(hybrid_result):
    _, r = hybrid_result
    assert r.matrix_game_solves == 0


def test_policy_iteration_matches_value_iteration():
    for game in (SMALL, SMALL_RANDOM):
        vi = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10).run()
        pi = NashQIteration(
            game, gamma=0.9, mode="hybrid", tol=1e-10
        ).run_policy_iteration(eval_sweeps=40)
        worst = max(abs(vi.values[s] - pi.values[s]) for s in vi.values)
        assert worst < 1e-7


def test_policy_iteration_pure_mode_uses_no_linear_program():
    pi = NashQIteration(SMALL, gamma=0.9, mode="pure").run_policy_iteration()
    assert pi.matrix_game_solves == 0


def test_policy_iteration_reports_staleness_that_decays_to_zero():
    pi = NashQIteration(SMALL, gamma=0.9, mode="hybrid", tol=1e-9).run_policy_iteration()
    trace = pi.staleness_trace
    assert trace is not None and len(trace) == pi.iterations
    assert trace[-1] < 1e-6  # frozen strategies match the final Nash
    assert max(trace) >= trace[-1]  # it was worse earlier


@pytest.mark.slow
def test_value_bracket_gaps_certify_the_solution():
    game = SoccerGame(move_order="random")
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9)
    r = solver.run()
    gaps = solver.value_bracket_gaps(r)
    assert len(gaps) == len(r.values)
    assert gaps.max() < 1e-6  # HiGHS solves each stage game to near machine eps


@pytest.mark.slow
def test_policy_iteration_solves_fewer_stage_games_on_the_random_game():
    game = SoccerGame(width=5, height=5, goal_rows=(1, 2, 3), move_order="random")
    vi = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-9).run()
    pi = NashQIteration(
        game, gamma=0.9, mode="hybrid", tol=1e-9
    ).run_policy_iteration(eval_sweeps=40)
    assert pi.matrix_game_solves < vi.matrix_game_solves
    assert max(abs(vi.values[s] - pi.values[s]) for s in vi.values) < 1e-6


# --------------------------------------------------------------- random game

SMALL_RANDOM = SoccerGame(width=4, height=3, goal_rows=(1,), move_order="random")


@pytest.fixture(scope="module")
def random_hybrid():
    game = SoccerGame(move_order="random")
    return game, NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-8).run()


def test_random_game_has_no_pure_stationary_equilibrium(random_hybrid):
    _, r = random_hybrid
    assert not r.pure_equilibrium_exists
    assert 0.90 < r.saddle_fraction < 1.0
    assert len(r.no_saddle_states) > 50


def test_no_saddle_states_come_in_mirror_pairs(random_hybrid):
    game, r = random_hybrid
    w = game.width - 1
    bad = set(r.no_saddle_states)
    for x0, y0, x1, y1, b in bad:
        assert (w - x1, y1, w - x0, y0, 1 - b) in bad


def test_random_kickoff_favours_the_initial_carrier(random_hybrid):
    game, r = random_hybrid
    assert r.values[game.initial_state()] > 0.05


def test_pure_mode_underestimates_random_kickoff(random_hybrid):
    game, hybrid = random_hybrid
    pure = NashQIteration(game, gamma=0.9, mode="pure", tol=1e-8).run()
    s0 = game.initial_state()
    assert pure.values[s0] <= hybrid.values[s0] - 0.05
    for s in hybrid.values:
        assert pure.values[s] <= hybrid.values[s] + 1e-6


def test_random_modes_agree_on_small_board():
    hybrid = NashQIteration(SMALL_RANDOM, gamma=0.9, mode="hybrid", tol=1e-9).run()
    mixed = NashQIteration(SMALL_RANDOM, gamma=0.9, mode="mixed", tol=1e-9).run()
    worst = max(abs(hybrid.values[s] - mixed.values[s]) for s in hybrid.values)
    assert worst < 1e-6


@pytest.mark.slow
def test_random_modes_agree_on_full_board(random_hybrid):
    game, hybrid = random_hybrid
    mixed = NashQIteration(game, gamma=0.9, mode="mixed", tol=1e-8).run()
    worst = max(abs(hybrid.values[s] - mixed.values[s]) for s in hybrid.values)
    assert worst < 1e-5
