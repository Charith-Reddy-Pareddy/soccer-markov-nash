"""End-to-end check of the A10 Part 2 pipeline: solve -> imitate -> roll out."""

import numpy as np

from soccer_nash.best_response import BestResponse
from soccer_nash.game import SoccerGame
from soccer_nash.mlp import MLP
from soccer_nash.opponents import part2_opponent
from soccer_nash.simulate import play_deterministic


def test_trained_network_produces_a_winning_q8_trajectory():
    game = SoccerGame()
    states = list(game.states())

    br = BestResponse(game, part2_opponent, me=0, gamma=0.9).solve()
    X = np.array(states, dtype=float)
    y = np.array([int(br.policy[s]) for s in states])

    net = MLP(h1=32, h2=32, seed=0)
    net.train(X, y, epochs=250, lr=5e-3, batch_size=256, seed=0)

    rollout = play_deterministic(
        game, net.policy_dict(states), part2_opponent, me=0
    )
    assert rollout.winner == 0
    assert rollout.steps <= game.max_steps
    assert rollout.trajectory[0] == game.initial_state()
    assert rollout.trajectory[-1] == (-1, -1, -1, -1, 0)

    # Q8 must be consistent with the network: each move is the net's argmax.
    policy = net.policy_dict(states)
    for state, (a0, _a1) in zip(rollout.trajectory, rollout.joint_actions):
        assert a0 == policy[state]


def test_weight_export_round_trips_through_a10_format():
    net = MLP(h1=16, h2=16, seed=0)
    text = net.to_a10()
    restored = MLP.from_a10(text)
    x = np.array([[0, 2, 6, 2, 0], [3, 3, 2, 1, 1]], dtype=float)
    assert np.array_equal(net.predict(x), restored.predict(x))
