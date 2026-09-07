import numpy as np
import pytest

from soccer_nash.mlp import MLP


def test_rejects_oversized_hidden_layers():
    with pytest.raises(ValueError):
        MLP(h1=100, h2=32)
    with pytest.raises(ValueError):
        MLP(h1=32, h2=128)


def test_probs_are_distributions():
    net = MLP(h1=16, h2=16, seed=1)
    x = np.array([[0, 2, 6, 2, 0], [3, 1, 4, 4, 1]], dtype=float)
    p = net.probs(x)
    assert p.shape == (2, 4)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


def test_a10_format_shapes():
    net = MLP(h1=20, h2=12, seed=2)
    blocks = net.to_a10().split("-----")
    assert len(blocks) == 3
    rows = [b.strip().splitlines() for b in blocks]
    assert len(rows[0]) == 5 and all(len(r.split(",")) == 20 for r in rows[0])
    assert len(rows[1]) == 20 and all(len(r.split(",")) == 12 for r in rows[1])
    assert len(rows[2]) == 12 and all(len(r.split(",")) == 4 for r in rows[2])


def test_a10_roundtrip_preserves_predictions():
    net = MLP(h1=24, h2=18, seed=3)
    x = np.array([[i % 7, (i * 3) % 5, (i * 2) % 7, i % 5, i % 2] for i in range(40)],
                 dtype=float)
    reloaded = MLP.from_a10(net.to_a10())
    assert np.array_equal(net.predict(x), reloaded.predict(x))
    assert np.allclose(net.probs(x), reloaded.probs(x), atol=1e-6)


def test_training_fits_a_small_map():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 7, size=(200, 5)).astype(float)
    # Two origin-through boundaries -> 4 classes; bias-free MLP handles these.
    y = (X[:, 0] > X[:, 2]).astype(int) + 2 * (X[:, 1] > X[:, 3]).astype(int)
    net = MLP(h1=64, h2=64, seed=0)
    hist = net.train(X, y, epochs=400, lr=5e-3, batch_size=32, seed=0)
    assert hist[-1] >= 0.97


def test_partial_labels_accept_any_masked_action():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 7, size=(200, 5)).astype(float)
    base = (X[:, 1] > X[:, 2]).astype(int) + 2 * (X[:, 0] > X[:, 4]).astype(int)
    mask = np.eye(4)[base]
    mask[:, 0] = 1.0  # action 0 is always acceptable too
    net = MLP(h1=48, h2=48, seed=1)
    hist = net.train(X, mask, epochs=300, lr=5e-3, batch_size=32, seed=1)
    assert hist[-1] >= 0.99
