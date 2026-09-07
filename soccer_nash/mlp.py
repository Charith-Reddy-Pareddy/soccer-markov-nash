"""Bias-free MLP in the A10 submission format.

5 inputs -> h1 -> h2 -> 4 outputs, ReLU on the hidden layers, softmax on the
output. No bias terms (the A10 weight format is three matrices only).

The A10 grader feeds raw integer states, so :meth:`predict` and
:meth:`to_a10` both operate on raw inputs. Training may use an internal input
scale for conditioning; :meth:`to_a10` folds it into the first matrix so the
exported weights behave identically on raw inputs.
"""

from __future__ import annotations

import numpy as np

_DEFAULT_SCALE = np.array([1 / 6, 1 / 4, 1 / 6, 1 / 4, 1.0])


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(
        self,
        h1: int = 64,
        h2: int = 64,
        seed: int = 0,
        scale: np.ndarray | None = None,
    ):
        if not (0 < h1 < 100 and 0 < h2 < 100):
            raise ValueError("A10 requires fewer than 100 units per hidden layer")
        rng = np.random.default_rng(seed)
        self.scale = _DEFAULT_SCALE.copy() if scale is None else np.asarray(scale, float)
        self.W1 = rng.normal(0, np.sqrt(2 / 5), (5, h1))
        self.W2 = rng.normal(0, np.sqrt(2 / h1), (h1, h2))
        self.W3 = rng.normal(0, np.sqrt(2 / h2), (h2, 4))

    # ----------------------------------------------------------------- forward
    def _forward(self, x_raw: np.ndarray):
        x = np.asarray(x_raw, dtype=float) * self.scale
        z1 = x @ self.W1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2
        a2 = np.maximum(z2, 0.0)
        p = _softmax(a2 @ self.W3)
        return p, (x, z1, a1, z2, a2)

    def probs(self, x_raw: np.ndarray) -> np.ndarray:
        return self._forward(np.atleast_2d(x_raw))[0]

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        return np.argmax(self.probs(x_raw), axis=1)

    # ---------------------------------------------------------------- training
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 400,
        lr: float = 3e-3,
        batch_size: int = 256,
        seed: int = 0,
        sample_weight: np.ndarray | None = None,
    ) -> list[float]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = len(X)
        # ``y`` is either integer labels or an (N, 4) mask of acceptable
        # actions (partial-label learning: any masked action counts as correct).
        if y.ndim == 2:
            target_mask = y.astype(float)
            hard_labels = np.argmax(target_mask, axis=1)
        else:
            hard_labels = y.astype(int)
            target_mask = np.eye(4)[hard_labels]
        w_all = (
            np.ones(n)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        rng = np.random.default_rng(seed)

        opt = {k: [np.zeros_like(w), np.zeros_like(w)] for k, w in self._named()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        history: list[float] = []
        t = 0

        for epoch in range(epochs):
            lr_e = lr * 0.5 * (1 + np.cos(np.pi * epoch / max(epochs - 1, 1)))
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                grads = self._grads(X[idx], target_mask[idx], w_all[idx])

                t += 1
                for (name, w), grad in zip(self._named(), grads):
                    m, v = opt[name]
                    m[:] = b1 * m + (1 - b1) * grad
                    v[:] = b2 * v + (1 - b2) * grad**2
                    mhat = m / (1 - b1**t)
                    vhat = v / (1 - b2**t)
                    w -= lr_e * mhat / (np.sqrt(vhat) + eps)

            preds = self.predict(X)
            history.append(float(np.mean(target_mask[np.arange(n), preds] > 0)))
        return history

    def _named(self):
        return (("W1", self.W1), ("W2", self.W2), ("W3", self.W3))

    def _grads(
        self, xb: np.ndarray, mask: np.ndarray, weight: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gradients of the (weighted) partial-label cross-entropy w.r.t. W1-3."""
        wb = np.asarray(weight, dtype=float)[:, None]
        p, (x, z1, a1, z2, a2) = self._forward(xb)

        # Soft target: probability mass renormalised onto the acceptable
        # actions; a one-entry mask reduces this to a plain one-hot label.
        masked = p * mask
        soft = masked / np.clip(masked.sum(axis=1, keepdims=True), 1e-12, None)
        delta = wb * (p - soft) / wb.sum()

        back2 = (delta @ self.W3.T) * (z2 > 0)
        back1 = (back2 @ self.W2.T) * (z1 > 0)

        return x.T @ back1, a1.T @ back2, a2.T @ delta

    # ------------------------------------------------------------------ export
    def to_a10(self) -> str:
        w1 = self.W1 * self.scale[:, None]
        blocks = [
            "\n".join(",".join(f"{v:.8g}" for v in row) for row in w)
            for w in (w1, self.W2, self.W3)
        ]
        return "\n-----\n".join(blocks)

    @classmethod
    def from_a10(cls, text: str) -> MLP:
        mats = [
            np.array(
                [[float(v) for v in line.split(",")] for line in block.strip().splitlines()]
            )
            for block in text.split("-----")
        ]
        w1, w2, w3 = mats
        net = cls(h1=w1.shape[1], h2=w2.shape[1], scale=np.ones(5))
        net.W1, net.W2, net.W3 = w1, w2, w3
        return net

    def policy_dict(self, states) -> dict[tuple, int]:
        states = list(states)
        preds = self.predict(np.array(states, dtype=float))
        return {s: int(a) for s, a in zip(states, preds)}
