"""Nash-DQN for the discounted finite-horizon soccer game: a network maps
``(state, remaining steps)`` to the 4x4 joint-action matrix ``Q``, trained by
fitted-Q updates whose targets use the *exact* transition expectations and the
minimax value of a target network's matrix at the next step (zero at the
horizon). The policies it reports are the minimax solution of its own matrix,
so it is scored exactly like the policy-gradient learners.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from soccer_nash.matrix_games import solve_zero_sum_batch
from soccer_nash.pg_finite import Net, features


def _transition_arrays(solver):
    game, states = solver.game, solver._states
    index = {s: i for i, s in enumerate(states)}
    n = solver._n
    k = max(len(solver._out[s][i, j]) for s in states for i in range(n) for j in range(n))
    nxt = np.zeros((len(states), n * n, k), dtype=int)
    prob = np.zeros((len(states), n * n, k))
    rew = np.zeros((len(states), n * n, k))
    term = np.ones((len(states), n * n, k), dtype=bool)
    for si, s in enumerate(states):
        for i in range(n):
            for j in range(n):
                for o, (p, ns, r0) in enumerate(solver._out[s][i, j]):
                    prob[si, i * n + j, o], rew[si, i * n + j, o] = p, r0
                    if not game.is_terminal(ns):
                        nxt[si, i * n + j, o], term[si, i * n + j, o] = index[ns], False
    return np.array(states, dtype=np.float32), nxt, prob, rew, term


def train_dqn(
    solver, gamma: float, horizon: int, steps: int = 4000, batch: int = 128,
    lr: float = 1e-3, sync: int = 100, seed: int = 0,
) -> Net:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    game = solver.game
    xs, nxt, prob, rew, term = _transition_arrays(solver)
    net = Net(game, 16)
    target = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for step in range(steps):
        si = rng.integers(len(xs), size=batch)
        t = rng.integers(horizon, size=batch)
        rem = ((horizon - t - 1) / horizon).astype(np.float32)
        nx = xs[nxt[si]]
        nfeat = np.concatenate(
            [nx, np.broadcast_to(rem[:, None, None, None], nx.shape[:3] + (1,))], axis=-1)
        with torch.no_grad():
            m = target(torch.from_numpy(nfeat.reshape(-1, 6))).numpy().reshape(-1, 4, 4)
        v = solve_zero_sum_batch(m)[0].reshape(batch, 16, -1)
        v[term[si]] = 0.0
        v[t + 1 == horizon] = 0.0
        y = (prob[si] * (rew[si] + gamma * v)).sum(-1)
        loss = torch.nn.functional.mse_loss(
            net(torch.cat([torch.from_numpy(xs[si]),
                           torch.from_numpy(1 - t[:, None] / horizon).float()], dim=1)),
            torch.tensor(y, dtype=torch.float32))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % sync == 0:
            target.load_state_dict(net.state_dict())
    return net


class QPolicies:
    """Minimax strategies of the network's matrix at every (step, state), solved
    lazily per step and cached; ``row`` / ``col`` are policies for ``finite_horizon``."""

    def __init__(self, solver, net: Net, horizon: int):
        self.solver, self.net, self.horizon = solver, net, horizon
        self.index = {s: i for i, s in enumerate(solver._states)}
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _solve(self, t: int):
        if t not in self._cache:
            states = self.solver._states
            with torch.no_grad():
                m = self.net(features(states, t, self.horizon)).numpy().reshape(-1, 4, 4)
            _, p, q = solve_zero_sum_batch(m)
            self._cache[t] = (p, q)
        return self._cache[t]

    def row(self, t, states):
        return self._solve(t)[0][[self.index[s] for s in states]]

    def col(self, t, states):
        return self._solve(t)[1][[self.index[s] for s in states]]
