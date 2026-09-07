import pytest

from soccer_nash.best_response import BestResponse, materialize
from soccer_nash.game import Action, SoccerGame
from soccer_nash.opponents import always_left, part2_opponent
from soccer_nash.simulate import play_deterministic


@pytest.fixture(scope="module")
def br_part2():
    game = SoccerGame()
    return game, BestResponse(game, part2_opponent, me=0, gamma=0.9).solve()


def test_rejects_random_move_order():
    with pytest.raises(ValueError):
        BestResponse(SoccerGame(move_order="random"), part2_opponent)


def test_policy_is_a_legal_action_everywhere(br_part2):
    _, r = br_part2
    assert set(r.policy.values()) <= set(Action)
    assert len(r.policy) == len(r.values)


def test_values_bounded(br_part2):
    _, r = br_part2
    assert all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in r.values.values())


def test_best_response_wins_from_kickoff(br_part2):
    game, r = br_part2
    rollout = play_deterministic(game, r.policy, part2_opponent, me=0)
    assert rollout.winner == 0
    assert rollout.steps <= game.max_steps


def test_kickoff_value_tracks_win_length(br_part2):
    game, r = br_part2
    rollout = play_deterministic(game, r.policy, part2_opponent, me=0)
    # Undiscounted +1 on the final transition, discounted once per prior step.
    assert r.values[game.initial_state()] == pytest.approx(0.9 ** (rollout.steps - 1))


def test_best_response_beats_always_left():
    game = SoccerGame()
    r = BestResponse(game, always_left, me=0, gamma=0.9).solve()
    rollout = play_deterministic(game, r.policy, always_left, me=0)
    assert rollout.winner == 0


def test_materialize_matches_callable():
    game = SoccerGame()
    mat = materialize(game, part2_opponent, me=1)
    for s in list(game.states())[::97]:
        assert mat[s] == part2_opponent(game, s, 1)
