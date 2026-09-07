import numpy as np

from soccer_nash.game import SoccerGame
from soccer_nash.geometry import FEATURE_NAMES, features
from soccer_nash.symmetry import mirror_state
from soccer_nash.tree import DecisionTree


def test_features_are_mirror_invariant():
    game = SoccerGame()
    for s in list(game.states())[::29]:
        f = features(game, s)
        fm = features(game, mirror_state(s, game.width))
        assert f == fm  # carrier-frame features do not change under the mirror


def test_defender_ahead_detects_the_standoff():
    game = SoccerGame()
    # carrier (player 0) at (3,2), defender directly in front at (4,2).
    f = features(game, (3, 2, 4, 2, 0))
    assert f["defender_ahead_same_row"] == 1
    assert f["defender_can_intercept"] == 1
    # defender behind the carrier
    f = features(game, (3, 2, 2, 2, 0))
    assert f["defender_ahead"] == 0


def test_carrier_can_score_next_flag():
    game = SoccerGame()
    f = features(game, (6, 2, 3, 3, 0))  # carrier on goal row at the right edge
    assert f["carrier_can_score_next"] == 1
    f = features(game, (6, 0, 3, 3, 0))  # right edge but not a goal row
    assert f["carrier_can_score_next"] == 0


def test_feature_names_match_dict_keys():
    game = SoccerGame()
    assert list(features(game, game.initial_state()).keys()) == FEATURE_NAMES


def test_decision_tree_learns_a_simple_and_rule():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(200, 3))
    y = ((X[:, 0] == 1) & (X[:, 1] == 1)).astype(int)
    tree = DecisionTree(max_depth=3, min_leaf=5).fit(X, y, ["a", "b", "c"])
    assert (tree.predict(X).astype(int) == y).mean() == 1.0
    assert any("-> 1" in r for r in tree.rules())
