"""Nash-Q with a neural stage-game estimate -- the "get DQN to replicate the
exact solver" step.

A small network maps a state to a 4x4 matrix ``Q(s, a0, a1)`` (player 0's
payoff). Fitted-Q iteration regresses it toward

    r(s, a0, a1) + gamma * val(Q_target(s'))

with a frozen target network (the same freezing idea as DQN and as
``run_policy_iteration``). ``val`` is the exact minimax of the tiny 4x4.

The point is not to beat the exact solver -- it is to measure how close a
function approximator gets on value error, action agreement and exploitability,
so that the eventual continuous-action work has a yardstick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soccer_nash.game import JOINT_ACTIONS, SoccerGame, State
from soccer_nash.matrix_games import game_value, pure_bounds

_SADDLE_TOL = 1e-7
_SCALE = np.array([1 / 6, 1 / 4, 1 / 6, 1 / 4, 1.0])


def _minimax(m: np.ndarray) -> float:
    lo, hi = pure_bounds(m)
    return lo if hi - lo <= _SADDLE_TOL else game_value(m)


class _QNet:
    """5 -> h -> h -> out ReLU regressor with biases (linear output), Adam.
    ``out=16`` for a 4x4 Q matrix, ``out=8`` for two 4-vectors of policy logits."""

    def __init__(self, h: int = 96, seed: int = 0, out: int = 16):
        rng = np.random.default_rng(seed)
        self.out = out
        self.W1 = rng.normal(0, np.sqrt(2 / 5), (5, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, np.sqrt(2 / h), (h, h))
        self.b2 = np.zeros(h)
        self.W3 = rng.normal(0, np.sqrt(2 / h), (h, out)) * 0.1
        self.b3 = np.zeros(out)
        self._opt = [[np.zeros_like(w), np.zeros_like(w)] for w in self._w()]
        self._t = 0

    def _w(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def _forward(self, X):
        x = np.asarray(X, float) * _SCALE
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        out = a2 @ self.W3 + self.b3
        return out, (x, z1, a1, z2, a2)

    def predict(self, X) -> np.ndarray:
        return self._forward(np.atleast_2d(X))[0]

    def matrix(self, state: State) -> np.ndarray:
        return self.predict(np.array(state, float))[0].reshape(4, 4)

    def policy(self, state: State) -> tuple[np.ndarray, np.ndarray]:
        """Two length-4 strategies from an ``out=8`` net (softmax per head)."""
        z = self.predict(np.array(state, float))[0]
        p = np.exp(z[:4] - z[:4].max())
        q = np.exp(z[4:] - z[4:].max())
        return p / p.sum(), q / q.sum()

    def step(self, X, target, lr=2e-3):
        out, (x, z1, a1, z2, a2) = self._forward(X)
        g = 2.0 * (out - target) / len(X)
        dW3 = a2.T @ g
        db3 = g.sum(0)
        g2 = (g @ self.W3.T) * (z2 > 0)
        dW2 = a1.T @ g2
        db2 = g2.sum(0)
        g1 = (g2 @ self.W2.T) * (z1 > 0)
        dW1 = x.T @ g1
        db1 = g1.sum(0)
        self._t += 1
        grads = (dW1, db1, dW2, db2, dW3, db3)
        for (w, (m, v), grad) in zip(self._w(), self._opt, grads):
            m[:] = 0.9 * m + 0.1 * grad
            v[:] = 0.999 * v + 0.001 * grad**2
            w -= lr * (m / (1 - 0.9**self._t)) / (np.sqrt(v / (1 - 0.999**self._t)) + 1e-8)
        return float(np.mean((out - target) ** 2))


@dataclass
class NashDQNResult:
    net: _QNet
    epochs: int
    loss_trace: list[float]


def train_nash_dqn(
    game: SoccerGame,
    gamma: float = 0.9,
    hidden: int = 64,
    epochs: int = 400,
    target_sync: int = 5,
    batch_size: int = 256,
    seed: int = 0,
) -> NashDQNResult:
    if game.move_order != "deterministic":
        raise ValueError("nash_dqn expects the deterministic game")
    states = list(game.states())
    X = np.array(states, float)
    n = len(states)

    # Precompute (next_state, r0) per (state, a0, a1).
    nxt = np.empty((n, 16), dtype=object)
    rew = np.zeros((n, 16))
    for si, s in enumerate(states):
        for k, (a0, a1) in enumerate(JOINT_ACTIONS):
            ns, (r0, _), _ = game.step(s, a0, a1)
            nxt[si, k] = ns
            rew[si, k] = r0

    index = {s: i for i, s in enumerate(states)}
    net = _QNet(hidden, seed)
    target = _QNet(hidden, seed)
    for tw, w in zip(target._w(), net._w()):
        tw[:] = w
    rng = np.random.default_rng(seed)
    losses: list[float] = []

    for epoch in range(epochs):
        # Bootstrap target: the maximin lower bound (vectorised, no LP). It
        # coincides with the minimax on saddle-point equilibria, so the
        # deterministic game's fixed point is unchanged.
        preds = target.predict(X).reshape(n, 4, 4)
        v_next = preds.min(axis=2).max(axis=1)
        # Build the regression target matrix for every state.
        tgt = np.zeros((n, 16))
        for k in range(16):
            for i in range(n):
                ns = nxt[i, k]
                cont = 0.0 if game.is_terminal(ns) else gamma * v_next[index[ns]]
                tgt[i, k] = rew[i, k] + cont

        perm = rng.permutation(n)
        ep_loss = 0.0
        for b in range(0, n, batch_size):
            idx = perm[b : b + batch_size]
            ep_loss += net.step(X[idx], tgt[idx]) * len(idx)
        losses.append(ep_loss / n)

        if (epoch + 1) % target_sync == 0:
            for tw, w in zip(target._w(), net._w()):
                tw[:] = w

    return NashDQNResult(net, epochs, losses)


def compare_to_exact(
    game: SoccerGame,
    net: _QNet,
    exact_values,
    exact_row_policy,
    gamma,
    exact_no_saddle: set | None = None,
):
    """Value error, action agreement, pure/mixed classification agreement, and
    exploitability of the DQN policy.

    ``exact_no_saddle`` is the exact solver's set of no-pure-saddle states; when
    given, ``classification_agreement`` reports how often the network agrees on
    whether a state's stage game has a pure saddle.
    """
    from soccer_nash.exploit import duality_gap
    from soccer_nash.matrix_games import solve_zero_sum

    exact_no_saddle = exact_no_saddle or set()
    states = list(game.states())
    verr = 0.0
    agree = 0
    class_agree = 0
    mean_verr = 0.0
    row_pol: dict[State, np.ndarray] = {}
    col_pol: dict[State, np.ndarray] = {}
    for s in states:
        m = net.matrix(s)
        err = abs(_minimax(m) - exact_values[s])
        verr = max(verr, err)
        mean_verr += err
        lo, hi = pure_bounds(m)
        net_mixed = hi - lo > _SADDLE_TOL
        if net_mixed == (s in exact_no_saddle):
            class_agree += 1
        if not net_mixed:
            p = np.zeros(4)
            q = np.zeros(4)
            p[int(np.argmax(m.min(axis=1)))] = 1.0
            q[int(np.argmin(m.max(axis=0)))] = 1.0
        else:
            _, p, q = solve_zero_sum(m)
        row_pol[s] = p
        col_pol[s] = q
        if np.argmax(p) == int(np.argmax(exact_row_policy[s])):
            agree += 1

    gap = duality_gap(game, row_pol, col_pol, gamma=gamma)
    n = len(states)
    return {
        "max_value_error": verr,
        "mean_value_error": mean_verr / n,
        "action_agreement": agree / n,
        "classification_agreement": class_agree / n,
        "duality_gap": gap,
    }


def train_policy_baseline(
    game: SoccerGame,
    exact_row_policy,
    exact_col_policy,
    hidden: int = 64,
    epochs: int = 400,
    lr: float = 3e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> _QNet:
    """A network trained *directly* on the exact equilibrium strategies -- the
    third baseline the meeting asked for: does the neural failure come from the
    value approximation or from extracting a policy out of it?"""
    if game.move_order != "deterministic":
        raise ValueError("nash_dqn expects the deterministic game")
    states = list(game.states())
    X = np.array(states, float)
    y = np.array([
        np.concatenate([exact_row_policy[s], exact_col_policy[s]]) for s in states
    ])
    net = _QNet(hidden, seed, out=8)
    rng = np.random.default_rng(seed)
    n = len(states)
    for epoch in range(epochs):
        lr_e = lr * 0.5 * (1 + np.cos(np.pi * epoch / max(epochs - 1, 1)))
        perm = rng.permutation(n)
        for b in range(0, n, batch_size):
            idx = perm[b : b + batch_size]
            net.step(X[idx], y[idx], lr=lr_e)
    return net


def compare_policy_to_exact(
    game: SoccerGame,
    net: _QNet,
    exact_matrix_of,
    exact_row_policy,
    exact_no_saddle: set | None = None,
    gamma: float = 0.9,
):
    """Score a policy network against the exact solver: how often it names the
    right action, whether its strategy is close to a Nash of the *exact* stage
    game (equilibrium regret), and its overall exploitability."""
    from soccer_nash.exploit import duality_gap
    from soccer_nash.numerics import epsilon_equilibrium

    exact_no_saddle = exact_no_saddle or set()
    states = list(game.states())
    agree = 0
    regret = 0.0
    row_pol: dict[State, np.ndarray] = {}
    col_pol: dict[State, np.ndarray] = {}
    for s in states:
        p, q = net.policy(s)
        row_pol[s], col_pol[s] = p, q
        if np.argmax(p) == int(np.argmax(exact_row_policy[s])):
            agree += 1
        regret = max(regret, epsilon_equilibrium(exact_matrix_of(s), p, q))
    gap = duality_gap(game, row_pol, col_pol, gamma=gamma)
    n = len(states)
    return {
        "action_agreement": agree / n,
        "max_equilibrium_regret": regret,
        "duality_gap": gap,
    }
