"""Exact (noise-free) gradient play on the stationary discounted soccer game.

The policy-gradient learners in `pg_finite.py` estimate gradients from sampled games. This module
removes the sampling: both players' values are computed exactly from the transition table (60
unrolled Bellman steps, so the discounted tail is below gamma^60), and each player follows the
exact gradient of its own objective, the value summed over every non-terminal state (as if each
game started anywhere). Two policy classes:

- ``table``: one row of logits per state, nothing shared between states;
- ``network``: a small network from the board (positions and ball) to logits, shared across states.

Progress is scored by the exact duality gap at each state: player 0's best-response value against
player 1's policy minus player 1's, which is 0 exactly at a Markov-perfect (stationary Nash)
equilibrium.
"""

from __future__ import annotations

import numpy as np
import torch

from .game import SoccerGame

UPDATES = ("adam", "sgd", "eg")


def tables(game: SoccerGame, states: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Next-state index, probability and player-0 reward for every (state, a0, a1, outcome), at
    most two outcomes (the random move order); index ``len(states)`` is the finished game."""
    index, n, a = {s: i for i, s in enumerate(states)}, len(states), game.n_actions
    nxt = np.full((n, a, a, 2), n)
    prob, rew = np.zeros((n, a, a, 2)), np.zeros((n, a, a, 2))
    for s in states:
        for i in range(a):
            for j in range(a):
                for k, (p, ns, (r0, _)) in enumerate(game.transitions(s, i, j)):
                    nxt[index[s], i, j, k] = index.get(ns, n)
                    prob[index[s], i, j, k], rew[index[s], i, j, k] = p, r0
    return nxt, prob, rew


def duality_gap(p0: np.ndarray, p1: np.ndarray, tab, gamma: float, iters: int = 400) -> np.ndarray:
    """Per state: player 0's best-response value against p1 minus player 1's against p0."""
    nxt, prob, rew = tab
    n = len(p0)
    v0, v1 = np.zeros(n + 1), np.zeros(n + 1)
    for _ in range(iters):
        q0 = (prob * (rew + gamma * v0[nxt])).sum(-1)
        q1 = (prob * (rew + gamma * v1[nxt])).sum(-1)
        v0[:-1] = np.einsum("sab,sb->sa", q0, p1).max(1)
        v1[:-1] = np.einsum("sa,sab->sb", p0, q1).min(1)
    return v0[:-1] - v1[:-1]


def _policies(game: SoccerGame, states: list, policy: str, init: float, hidden: int):
    """Parameters per player and a function from player to its (states, actions) probabilities."""
    n, a = len(states), game.n_actions
    if policy == "table":
        logits = [(init * torch.randn(n, a)).requires_grad_() for _ in range(2)]
        return [[t] for t in logits], lambda i: torch.softmax(logits[i], 1)
    w, h = game.width, game.height
    feats = torch.tensor([[s[0] / (w - 1), s[1] / (h - 1), s[2] / (w - 1), s[3] / (h - 1), s[4]]
                          for s in states], dtype=torch.float32)
    nets = [torch.nn.Sequential(torch.nn.Linear(5, hidden), torch.nn.Tanh(),
                                torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                                torch.nn.Linear(hidden, a)) for _ in range(2)]
    return [list(m.parameters()) for m in nets], lambda i: torch.softmax(nets[i](feats), 1)


def play(game: SoccerGame, policy: str = "table", method: str = "adam", steps: int = 2000,
         lr: float = 0.05, seed: int = 0, gamma: float = 0.9, rounds: int = 60, init: float = 0.0,
         every: int = 100, hidden: int = 64):
    """Run exact gradient play. ``method`` is ``adam`` (simultaneous Adam on the mean value),
    ``sgd`` (simultaneous gradient steps on the summed value) or ``eg`` (extragradient on the
    summed value). Table logits start at ``init`` times a standard normal draw. Returns rows
    (step, mean gap, max gap, kickoff gap)."""
    if method not in UPDATES:
        raise ValueError(f"unknown method {method!r}")
    states = list(game.states())
    tab = tables(game, states)
    nxt = torch.as_tensor(tab[0])
    prob, rew = (torch.as_tensor(t, dtype=torch.float32) for t in tab[1:])
    n, kick = len(states), states.index(game.initial_state())
    torch.manual_seed(seed)
    params, probs = _policies(game, states, policy, init, hidden)

    def grads():
        v, p0, p1 = torch.zeros(n + 1), probs(0), probs(1)
        for _ in range(rounds):
            q = (prob * (rew + gamma * v[nxt])).sum(-1)
            v = torch.cat([torch.einsum("sa,sab,sb->s", p0, q, p1), torch.zeros(1)])
        objective = v[:-1].mean() if method == "adam" else v[:-1].sum()
        return (torch.autograd.grad(-objective, params[0], retain_graph=True),
                torch.autograd.grad(objective, params[1]))

    def step(gs, size):
        with torch.no_grad():
            for ps, gr in zip(params, gs):
                for p, g in zip(ps, gr):
                    p -= size * g

    opts = [torch.optim.Adam(p, lr=lr) for p in params] if method == "adam" else None
    rows = []
    for t in range(steps + 1):
        if t % every == 0:
            p0, p1 = (probs(i).detach().numpy().astype(float) for i in (0, 1))
            gap = duality_gap(p0, p1, tab, gamma)
            rows.append((t, float(gap.mean()), float(gap.max()), float(gap[kick])))
        if t == steps:
            break
        gs = grads()
        if method == "adam":
            for opt, ps, gr in zip(opts, params, gs):
                for p, g in zip(ps, gr):
                    p.grad = g
                opt.step()
        elif method == "sgd":
            step(gs, lr)
        else:  # extragradient: a look-ahead step, then the step from the start with its gradient
            saved = [[p.detach().clone() for p in ps] for ps in params]
            step(gs, lr)
            look = grads()
            with torch.no_grad():
                for ps, sv in zip(params, saved):
                    for p, v in zip(ps, sv):
                        p.copy_(v)
            step(look, lr)
    return rows
