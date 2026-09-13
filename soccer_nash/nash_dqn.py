"""Nash-Q with a neural stage-game estimate -- the "get DQN to replicate the
exact solver" step.

A small network maps a state to a 4x4 matrix ``Q(s, a0, a1)`` (player 0's
payoff). Fitted-Q iteration regresses it toward

    r(s, a0, a1) + gamma * val(Q_target(s'))

with a frozen target network (the same freezing idea as DQN and as
``run_policy_iteration``). ``val`` is the exact minimax of the tiny 4x4 --
the pure-saddle fast path plus an LP fallback (:func:`_minimax_batch`), not
a maximin-only shortcut, so this is exact on *any* move order, including
ones whose stage games are genuinely mixed (``scripts/nash_dqn_random.py`).
The transition precompute uses ``game.transitions`` (the full outcome
distribution), not ``game.step`` (which samples one outcome) -- the two only
coincide when every joint action has a single outcome, i.e. only on the
deterministic game.

The point is not to beat the exact solver -- it is to measure how close a
function approximator gets on value error, action agreement and exploitability,
so that the eventual continuous-action work has a yardstick.

Three starting points for the Q-net, all comparable via :func:`compare_to_exact`:

- :func:`train_nash_dqn` -- **from zero**: random init, trained purely by TD
  bootstrap against a frozen target network.
- :func:`fit_q_to_exact` -- **fit to exact**: supervised regression straight
  onto ``Q_exact``, no bootstrap at all -- how well the network can represent
  the answer when it is simply told what it is.
- :func:`train_nash_dqn` again, passed ``init_net=`` the net
  :func:`fit_q_to_exact` returned -- **warm start**: the same TD bootstrap as
  "from zero", but starting already at (an approximation of) the exact
  solution, to see whether the bootstrap objective holds that starting point
  or pulls the network away from it.
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


def _minimax_batch(M: np.ndarray) -> np.ndarray:
    """The exact hybrid minimax value of every matrix in a batch
    ``M`` (shape ``(n, 4, 4)``) -- the pure-saddle check (``pure_bounds``,
    vectorised, no LP) for every matrix at once, then one LP
    (:func:`soccer_nash.matrix_games.game_value`) only for the matrices that
    don't have one. This is what makes fitted-Q iteration correct on a game
    whose stage matrices are sometimes genuinely mixed (``move_order=
    "random"``/``"coinflip"``/``"tackle"``/``"blend"``) rather than the
    pure-maximin-only shortcut that is exact solely on the ``deterministic``
    game, where every stage game has a pure saddle."""
    col_min = M.min(axis=2)
    lo = col_min.max(axis=1)
    row_max = M.max(axis=1)
    hi = row_max.min(axis=1)
    v = lo.copy()
    mixed_idx = np.flatnonzero(hi - lo > _SADDLE_TOL)
    for i in mixed_idx:
        v[i] = game_value(M[i])
    return v


class _QNet:
    """5 -> hidden layers -> out ReLU regressor with biases (linear output
    layer), Adam. ``hidden`` is either a single int -- 2 hidden layers of
    that width, the original architecture -- or a tuple of ints, one entry
    per hidden layer, for the width/depth ablation
    (``scripts/nash_dqn_ablation.py``): ``(256,)`` is one wide layer,
    ``(64, 64, 64)`` is three narrower ones, etc. ``out=16`` for a 4x4 Q
    matrix, ``out=8`` for two 4-vectors of policy logits.

    Passing an int for ``hidden`` reproduces the original 2-hidden-layer
    network bit-for-bit (same init formula, same weight order, same seed ->
    same weights) -- this class only generalizes *how many* hidden layers
    there are, not the per-layer init scheme or the training math.
    """

    def __init__(self, h: int | tuple[int, ...] = 96, seed: int = 0, out: int = 16):
        hidden = (h, h) if isinstance(h, int) else tuple(h)
        rng = np.random.default_rng(seed)
        self.out = out
        self.hidden = hidden
        sizes = [5, *hidden, out]
        n_layers = len(sizes) - 1  # weight matrices, not hidden layers
        self.Ws: list[np.ndarray] = []
        self.bs: list[np.ndarray] = []
        for i in range(n_layers):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            W = rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))
            if i == n_layers - 1:
                W = W * 0.1  # output layer: small init, as in the original
            self.Ws.append(W)
            self.bs.append(np.zeros(fan_out))
        self._opt = [[np.zeros_like(w), np.zeros_like(w)] for w in self._w()]
        self._t = 0

    def _w(self):
        out = []
        for W, b in zip(self.Ws, self.bs):
            out.append(W)
            out.append(b)
        return out

    def _forward(self, X):
        x = np.asarray(X, float) * _SCALE
        n_layers = len(self.Ws)
        acts = [x]  # acts[i] feeds Ws[i]; acts[-1] is the (linear) output
        pre: list[np.ndarray] = []
        a = x
        for i in range(n_layers):
            z = a @ self.Ws[i] + self.bs[i]
            pre.append(z)
            a = np.maximum(z, 0) if i < n_layers - 1 else z
            acts.append(a)
        return a, (acts, pre)

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
        out, (acts, pre) = self._forward(X)
        n_layers = len(self.Ws)
        g = 2.0 * (out - target) / len(X)
        grads: list[np.ndarray | None] = [None] * (2 * n_layers)
        for i in reversed(range(n_layers)):
            grads[2 * i] = acts[i].T @ g
            grads[2 * i + 1] = g.sum(0)
            if i > 0:
                g = (g @ self.Ws[i].T) * (pre[i - 1] > 0)
        self._t += 1
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


def _precompute_transitions(game: SoccerGame, states: list[State]):
    """``(probs, rews, nidx, term)``, each shape ``(n, 16, max_outcomes)`` --
    the exact transition distribution of every joint action at every state
    (``game.transitions``, not ``game.step``: ``step`` *samples* one outcome,
    which is only equivalent to the exact expectation on the deterministic
    game where every joint action already has a single outcome. Any other
    move order needs the real distribution averaged, not one sampled draw).
    ``nidx`` indexes into ``states``; ``term`` marks a terminal (or padding)
    outcome, for which the continuation value is 0 regardless of ``nidx``."""
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    raw = [
        [game.transitions(s, a0, a1) for a0, a1 in JOINT_ACTIONS] for s in states
    ]
    max_out = max(len(outs) for row in raw for outs in row)
    probs = np.zeros((n, 16, max_out))
    rews = np.zeros((n, 16, max_out))
    nidx = np.zeros((n, 16, max_out), dtype=int)
    term = np.ones((n, 16, max_out), dtype=bool)
    for si, row in enumerate(raw):
        for k, outs in enumerate(row):
            for oi, (p, ns, (r0, _r1)) in enumerate(outs):
                probs[si, k, oi] = p
                rews[si, k, oi] = r0
                if not game.is_terminal(ns):
                    nidx[si, k, oi] = index[ns]
                    term[si, k, oi] = False
    return probs, rews, nidx, term


def train_nash_dqn(
    game: SoccerGame,
    gamma: float = 0.9,
    hidden: int = 64,
    epochs: int = 400,
    target_sync: int = 5,
    batch_size: int = 256,
    seed: int = 0,
    init_net: _QNet | None = None,
) -> NashDQNResult:
    """Fitted-Q / DQN-style training, exact for *any* move order.

    The bootstrap target at every ``(s, a0, a1)`` is the full expectation
    over ``game.transitions`` -- ``sum_outcomes prob * (r + gamma *
    minimax(Q_target(s')))`` -- using :func:`_minimax_batch`'s pure-fast-path
    + LP hybrid for ``minimax``, not the maximin-only shortcut that is exact
    solely when every stage game already has a pure saddle. On the
    deterministic game this reduces to exactly the old behaviour (one
    outcome per joint action, pure fast path every time); on ``"random"``/
    ``"coinflip"``/``"tackle"``/``"blend"`` it is now the correct target
    where a stage game is genuinely mixed.

    ``init_net``, when given, seeds both the online and target network's
    weights from it instead of the usual random (He-normal) init -- e.g. the
    output of :func:`fit_q_to_exact`, to test whether starting the TD
    bootstrap already at (an approximation of) the exact solution changes
    where it ends up, versus starting from scratch (``init_net=None``). It
    must share this call's ``hidden``/``out`` shape.
    """
    states = list(game.states())
    X = np.array(states, float)
    n = len(states)

    probs, rews, nidx, term = _precompute_transitions(game, states)

    net = _QNet(hidden, seed)
    if init_net is not None:
        for w, iw in zip(net._w(), init_net._w()):
            w[:] = iw
    target = _QNet(hidden, seed)
    for tw, w in zip(target._w(), net._w()):
        tw[:] = w
    rng = np.random.default_rng(seed)
    losses: list[float] = []

    for epoch in range(epochs):
        preds = target.predict(X).reshape(n, 4, 4)
        v_next = _minimax_batch(preds)
        cont = np.where(term, 0.0, gamma * v_next[nidx])
        tgt = (probs * (rews + cont)).sum(axis=2)  # shape (n, 16)

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


def _support_set(p: np.ndarray, tol: float = 1e-6) -> frozenset[int]:
    """Actions with positive probability -- same convention as
    ``soccer_nash.numerics.support_shape`` and ``scripts/positions.py``'s
    ``_support``, so support-agreement here is comparable to the canonical
    (2,2)/(3,3)/(2,1)/(3,2) shape counts elsewhere in the project."""
    return frozenset(i for i, v in enumerate(p) if v > tol)


def compare_to_exact(
    game: SoccerGame,
    net: _QNet,
    exact_values,
    exact_row_policy,
    gamma,
    exact_no_saddle: set | None = None,
    exact_matrix_of=None,
    exact_col_policy=None,
):
    """Value error, action agreement, pure/mixed classification agreement, and
    exploitability of the DQN policy -- plus, when ``exact_matrix_of`` is
    given, the matrix-level metrics that ``action_agreement`` alone can hide
    in a game with ties and non-unique equilibria: mean/max entrywise
    (Frobenius) error against ``Q_exact``, the extracted policy's exact
    equilibrium regret (:func:`soccer_nash.numerics.epsilon_equilibrium`
    against the *exact* stage matrix, not the net's own predicted one), and
    support agreement (does the net's policy use the same action set as the
    exact one, ``exact_col_policy`` needed for the defender side).

    ``exact_no_saddle`` is the exact solver's set of no-pure-saddle states; when
    given, ``classification_agreement`` reports how often the network agrees on
    whether a state's stage game has a pure saddle.
    """
    from soccer_nash.exploit import duality_gap
    from soccer_nash.matrix_games import solve_zero_sum
    from soccer_nash.numerics import epsilon_equilibrium

    exact_no_saddle = exact_no_saddle or set()
    states = list(game.states())
    verr = 0.0
    agree = 0
    class_agree = 0
    mean_verr = 0.0
    frob_sum = 0.0
    frob_max = 0.0
    entry_max = 0.0
    regret_sum = 0.0
    regret_max = 0.0
    row_support_agree = 0
    col_support_agree = 0
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

        if exact_matrix_of is not None:
            m_exact = exact_matrix_of(s)
            diff = m - m_exact
            frob = float(np.sqrt((diff**2).sum()))
            frob_sum += frob
            frob_max = max(frob_max, frob)
            entry_max = max(entry_max, float(np.abs(diff).max()))
            reg = float(epsilon_equilibrium(m_exact, p, q))
            regret_sum += reg
            regret_max = max(regret_max, reg)
            if _support_set(p) == _support_set(exact_row_policy[s]):
                row_support_agree += 1
            if exact_col_policy is not None and _support_set(q) == _support_set(
                exact_col_policy[s]
            ):
                col_support_agree += 1

    gap = duality_gap(game, row_pol, col_pol, gamma=gamma)
    n = len(states)
    out = {
        "max_value_error": verr,
        "mean_value_error": mean_verr / n,
        "action_agreement": agree / n,
        "classification_agreement": class_agree / n,
        "duality_gap": gap,
    }
    if exact_matrix_of is not None:
        out.update({
            "mean_frobenius_error": frob_sum / n,
            "max_frobenius_error": frob_max,
            "max_entrywise_error": entry_max,
            "mean_equilibrium_regret": regret_sum / n,
            "max_equilibrium_regret": regret_max,
            "row_support_agreement": row_support_agree / n,
        })
        if exact_col_policy is not None:
            out["col_support_agreement"] = col_support_agree / n
    return out


def fit_q_to_exact(
    game: SoccerGame,
    exact_matrix_of,
    hidden: int = 64,
    epochs: int = 400,
    lr: float = 3e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> _QNet:
    """A Q-net trained *directly* on the exact stage-game matrices --
    supervised regression, no TD bootstrap, no target network. This is the
    "fit a DQN to the exact solution" step: it isolates how well a network of
    this size can even *represent* ``Q_exact``, before any bootstrapping
    noise gets involved. Mirrors :func:`train_policy_baseline`, but for the
    Q-matrix head (``out=16``) instead of the two policy heads (``out=8``).
    Its returned net is a valid ``init_net`` for :func:`train_nash_dqn`. Any
    move order works -- this function only ever sees ``exact_matrix_of``'s
    output, already the correct stage matrix (pure or mixed) for whichever
    game produced it.
    """
    states = list(game.states())
    X = np.array(states, float)
    y = np.array([exact_matrix_of(s).reshape(-1) for s in states])
    net = _QNet(hidden, seed, out=16)
    rng = np.random.default_rng(seed)
    n = len(states)
    for epoch in range(epochs):
        lr_e = lr * 0.5 * (1 + np.cos(np.pi * epoch / max(epochs - 1, 1)))
        perm = rng.permutation(n)
        for b in range(0, n, batch_size):
            idx = perm[b : b + batch_size]
            net.step(X[idx], y[idx], lr=lr_e)
    return net


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
    value approximation or from extracting a policy out of it? Any move order
    works -- this only ever sees the already-computed exact policies."""
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
