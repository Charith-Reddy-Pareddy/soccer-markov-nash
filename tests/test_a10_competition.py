import numpy as np
import pytest

from soccer_nash.exploit import best_response_to, onehot_policy
from soccer_nash.game import SoccerGame
from soccer_nash.mlp import MLP
from soccer_nash.nash_q import NashQIteration


@pytest.fixture(scope="module")
def solved():
    game = SoccerGame()
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    result = solver.run()
    states, row_mask, col_mask = solver.optimal_action_masks(result)
    return game, solver, result, states, row_mask, col_mask


def test_masks_are_binary_and_nonempty(solved):
    _, _, _, states, row_mask, col_mask = solved
    assert row_mask.shape == (len(states), 4)
    assert set(np.unique(row_mask)) <= {0.0, 1.0}
    assert (row_mask.sum(axis=1) >= 1).all()
    assert (col_mask.sum(axis=1) >= 1).all()


def test_nash_policy_support_is_inside_the_mask(solved):
    _, _, result, states, row_mask, col_mask = solved
    for i, s in enumerate(states):
        assert row_mask[i, np.argmax(result.row_policy[s])] == 1.0
        assert col_mask[i, np.argmax(result.col_policy[s])] == 1.0


def test_masks_are_mirror_symmetric(solved):
    game, _, _, states, row_mask, col_mask = solved
    w = game.width - 1
    flip = {2: 3, 3: 2, 0: 0, 1: 1}  # L<->R under the board mirror
    index = {s: i for i, s in enumerate(states)}
    for i, (x0, y0, x1, y1, b) in enumerate(states):
        j = index[(w - x1, y1, w - x0, y0, 1 - b)]
        for a in range(4):
            assert row_mask[i, a] == col_mask[j, flip[a]]


def test_trained_networks_are_close_to_unexploitable(solved):
    game, solver, result, states, row_mask, col_mask = solved
    X = np.array(states, dtype=float)
    s0 = game.initial_state()

    net0 = MLP(h1=48, h2=48, seed=0)
    net0.train(X, row_mask, epochs=500, lr=4e-3, batch_size=256, seed=0)
    net1 = MLP(h1=48, h2=48, seed=1)
    net1.train(X, col_mask, epochs=500, lr=4e-3, batch_size=256, seed=1)

    v1 = best_response_to(
        game, onehot_policy(net0.policy_dict(states)), responder=1, gamma=0.9
    ).values[s0]
    # equilibrium value is 0; a decent net keeps player 1 well under a sure win.
    assert v1 + result.values[s0] < 0.5
