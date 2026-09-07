"""Can the pure / mixed classification of a soccer stage game be predicted from
the spatial geometry of the state alone?

Builds a feature table for the random-move-order game, fits a small decision
tree, and reports the rule.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from soccer_nash.game import SoccerGame
from soccer_nash.geometry import FEATURE_NAMES, features
from soccer_nash.nash_q import NashQIteration
from soccer_nash.numerics import classify_stage_game, support_shape
from soccer_nash.tree import DecisionTree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    args = parser.parse_args()

    game = SoccerGame(move_order="random")
    solver = NashQIteration(game, gamma=args.gamma, mode="hybrid", tol=1e-10)
    result = solver.run()

    X, y_type = [], []
    for s in solver._states:
        X.append([features(game, s)[k] for k in FEATURE_NAMES])
        y_type.append(classify_stage_game(solver._matrix(s, result.values)))
    X = np.array(X)
    y_type = np.array(y_type, dtype=object)
    y_mixed = (y_type == "mixed").astype(int)

    print(f"class balance: {dict(collections.Counter(y_type))}\n")

    # Single-feature separation for "mixed".
    print("feature -> P(mixed | feature) vs P(mixed | not feature):")
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        if set(np.unique(col)) <= {0, 1}:
            hi = y_mixed[col == 1].mean() if (col == 1).any() else 0
            lo = y_mixed[col == 0].mean() if (col == 0).any() else 0
            if abs(hi - lo) > 0.02:
                print(f"  {name:<26s} {hi:.3f}  vs  {lo:.3f}")

    tree = DecisionTree(max_depth=4, min_leaf=8).fit(X, y_mixed, FEATURE_NAMES)
    pred = tree.predict(X).astype(int)
    tp = int(((pred == 1) & (y_mixed == 1)).sum())
    fp = int(((pred == 1) & (y_mixed == 0)).sum())
    fn = int(((pred == 0) & (y_mixed == 1)).sum())
    print(f"\ndecision tree for 'mixed':  precision {tp / max(tp + fp, 1):.2f}, "
          f"recall {tp / max(tp + fn, 1):.2f}")
    for rule in tree.rules():
        if "-> 1" in rule:
            print("  " + rule)

    # Template clustering of the mixed states by essential support and geometry.
    print("\nmixed-state templates (support shape x defender geometry):")
    templates = collections.Counter()
    for s in result.no_saddle_states:
        m = solver._matrix(s, result.values)
        f = features(game, s)
        geom = (
            "def-ahead-same-row"
            if f["defender_ahead_same_row"]
            else "def-ahead" if f["defender_ahead"]
            else "adjacent" if f["adjacent"]
            else "other"
        )
        templates[(support_shape(m), geom)] += 1
    for (shape, geom), n in templates.most_common():
        print(f"  support {shape}  {geom:<20s}  {n}")


if __name__ == "__main__":
    main()
