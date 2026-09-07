"""A tiny greedy decision tree for interpretable classification (numpy only).

Just enough to ask "can this label be predicted from these integer features,
and what is the rule" -- not a general ML library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - float((p * p).sum())


@dataclass
class Node:
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    label: object = None
    n: int = 0
    impurity: float = 0.0


class DecisionTree:
    def __init__(self, max_depth: int = 4, min_leaf: int = 5):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.root: Node | None = None
        self.feature_names: list[str] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        X = np.asarray(X)
        y = np.asarray(y)
        self.feature_names = feature_names
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        vals, counts = np.unique(y, return_counts=True)
        majority = vals[int(np.argmax(counts))]
        node = Node(label=majority, n=len(y), impurity=_gini(y))

        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or len(vals) == 1:
            return node

        best_gain, best = 0.0, None
        parent = _gini(y)
        for f in range(X.shape[1]):
            for thr in np.unique(X[:, f])[:-1]:
                mask = X[:, f] <= thr
                if mask.sum() < self.min_leaf or (~mask).sum() < self.min_leaf:
                    continue
                w = mask.mean()
                gain = parent - w * _gini(y[mask]) - (1 - w) * _gini(y[~mask])
                if gain > best_gain:
                    best_gain, best = gain, (f, float(thr), mask)

        if best is None:
            return node
        f, thr, mask = best
        node.feature, node.threshold = f, thr
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.array([self._predict_one(row, self.root) for row in X], dtype=object)

    def _predict_one(self, row, node: Node):
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.label

    def rules(self) -> list[str]:
        names = self.feature_names or [f"x{i}" for i in range(999)]
        out: list[str] = []

        def walk(node: Node, cond: list[str]) -> None:
            if node.feature is None:
                prefix = " and ".join(cond) if cond else "always"
                out.append(f"IF {prefix} -> {node.label}  (n={node.n})")
                return
            f = names[node.feature]
            thr = node.threshold
            test = f"{f}<={thr:g}" if thr != 0 else f"not {f}"
            walk(node.left, cond + [test])
            test = f"{f}>{thr:g}" if thr != 0 else f"{f}"
            walk(node.right, cond + [test])

        walk(self.root, [])
        return out
