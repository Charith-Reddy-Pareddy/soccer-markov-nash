"""REINFORCE, A2C and PPO for the discounted finite-horizon soccer game, trained
either by plain self-play or by fictitious play: policy gradient solves the game
by itself, with no exact stage-game solving, and sees the remaining step count as
an input.

Fictitious play here is the training scheme, not a matrix solver: each player
keeps improving its network by policy gradient against the *empirical average*
of the opponent's past policies (frozen snapshots, one drawn uniformly per
episode), and the policy it reports is the average of its own snapshots.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from soccer_nash.game import MOVE_ACTIONS, SoccerGame, State

ALGOS = ("reinforce", "a2c", "ppo")


def features(states: list[State], t: int, horizon: int) -> torch.Tensor:
    """State plus the fraction of the horizon still remaining."""
    x = torch.tensor(np.array(states, dtype=np.float32))
    return torch.cat([x, torch.full((len(states), 1), (horizon - t) / horizon)], dim=1)


class Net(nn.Module):
    def __init__(self, game: SoccerGame, n_out: int, hidden: int = 64):
        super().__init__()
        w, h = game.width - 1, game.height - 1
        self.register_buffer("scale", torch.tensor([1 / w, 1 / h, 1 / w, 1 / h, 1.0, 1.0]))
        self.body = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x * self.scale)


def _frozen(net: Net) -> Net:
    return copy.deepcopy(net).requires_grad_(False)


class _Current:
    def __init__(self, net: Net):
        self.net = net

    def probs(self, x: torch.Tensor, ep: np.ndarray) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class _Mixture:
    """Each episode plays one snapshot, drawn uniformly -- the empirical average."""
    def __init__(self, snaps: list[Net], n_episodes: int, rng: np.random.Generator):
        self.snaps = snaps
        self.pick = rng.integers(len(snaps), size=n_episodes)

    def probs(self, x: torch.Tensor, ep: np.ndarray) -> torch.Tensor:
        out = torch.zeros(len(ep), 4)
        which = self.pick[ep]
        for k in np.unique(which):
            m = torch.from_numpy(which == k)
            out[m] = torch.softmax(self.snaps[int(k)](x[m]), dim=-1)
        return out


def collect(game, p0, p1, n_ep: int, horizon: int, rng) -> dict:
    states = [game.initial_state()] * n_ep
    alive = np.ones(n_ep, dtype=bool)
    X = torch.zeros(horizon, n_ep, 6)
    A = torch.zeros(2, horizon, n_ep, dtype=torch.long)
    LP = torch.zeros(2, horizon, n_ep)
    R = torch.zeros(horizon, n_ep)
    ALIVE = torch.zeros(horizon, n_ep, dtype=torch.bool)
    for t in range(horizon):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        x = features([states[i] for i in idx], t, horizon)
        with torch.no_grad():
            pr = [p.probs(x, idx) for p in (p0, p1)]
            act = [torch.multinomial(q, 1).squeeze(1) for q in pr]
        ti = torch.from_numpy(idx)
        X[t, ti] = x
        ALIVE[t, ti] = True
        for k in range(2):
            A[k, t, ti] = act[k]
            LP[k, t, ti] = pr[k].gather(1, act[k][:, None]).squeeze(1).log()
        for n, i in enumerate(idx):
            states[i], (r0, _), done = game.step(
                states[i], MOVE_ACTIONS[int(act[0][n])], MOVE_ACTIONS[int(act[1][n])], rng=rng)
            R[t, i] = r0
            if done:
                alive[i] = False
    return {"X": X, "A": A, "LP": LP, "R": R, "alive": ALIVE}


def _returns(r, alive, value, algo, gamma, n_step=10, lam=0.95):
    """Discounted return / advantage targets for a (T, E) batch."""
    T = r.shape[0]
    nxt = torch.cat([alive[1:], torch.zeros(1, r.shape[1], dtype=torch.bool)]).float()
    if algo == "reinforce":
        g, out = torch.zeros(r.shape[1]), torch.zeros_like(r)
        for t in reversed(range(T)):
            g = r[t] + gamma * g * nxt[t]
            out[t] = g
        return out, None
    v_next = torch.cat([value[1:], torch.zeros(1, r.shape[1])]) * nxt
    if algo == "a2c":
        out = torch.zeros_like(r)
        for t in range(T):
            end = min(t + n_step, T)
            ret = (value[end] * alive[end].float()) if end < T else torch.zeros(r.shape[1])
            for k in reversed(range(t, end)):
                ret = r[k] + gamma * ret * nxt[k]
            out[t] = ret
        return out, out
    adv, g = torch.zeros_like(r), torch.zeros(r.shape[1])
    for t in reversed(range(T)):
        delta = r[t] + gamma * v_next[t] - value[t]
        g = delta + gamma * lam * nxt[t] * g
        adv[t] = g
    return adv, adv + value


def update(net, critic, opts, batch, player, algo, gamma, entropy, clip=0.2, epochs=4):
    sign = 1.0 if player == 0 else -1.0
    X, alive = batch["X"], batch["alive"]
    r = sign * batch["R"]
    with torch.no_grad():
        value = critic(X).squeeze(-1) if critic is not None else None
    adv, target = _returns(r, alive, value, algo, gamma)
    x, a = X[alive], batch["A"][player][alive]
    old, adv_f = batch["LP"][player][alive], adv[alive]
    if algo == "ppo":
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
    elif algo == "a2c":
        adv_f = adv_f - value[alive]
    for _ in range(epochs if algo == "ppo" else 1):
        logp_all = torch.log_softmax(net(x), dim=-1)
        logp = logp_all.gather(1, a[:, None]).squeeze(1)
        if algo == "ppo":
            ratio = (logp - old).exp()
            pl = -torch.min(ratio * adv_f, ratio.clamp(1 - clip, 1 + clip) * adv_f).mean()
        else:
            pl = -(logp * adv_f).mean()
        ent = -(logp_all.exp() * logp_all).sum(-1).mean()
        loss = pl - entropy * ent
        if critic is not None:
            loss = loss + 0.5 * nn.functional.mse_loss(
                critic(x).squeeze(-1), target[alive])
        opts[player].zero_grad()
        loss.backward()
        opts[player].step()


@dataclass
class Trained:
    """Reported policies: ``pol0`` / ``pol1(t, states) -> (n, 4)``."""
    pol0: object
    pol1: object


def _policy(nets: list[Net], horizon: int):
    def pol(t, states):
        with torch.no_grad():
            x = features(list(states), t, horizon)
            return torch.stack([torch.softmax(n(x), dim=-1) for n in nets]).mean(0).numpy()
    return pol


def train(
    game: SoccerGame, algo: str, mode: str = "selfplay", gamma: float = 0.9,
    horizon: int = 100, iterations: int = 300, episodes: int = 64, lr: float = 1e-3,
    entropy: float = 0.01, snap_every: int = 5, seed: int = 0,
) -> Trained:
    if algo not in ALGOS or mode not in ("selfplay", "fictitious"):
        raise ValueError("algo must be one of ALGOS, mode 'selfplay' or 'fictitious'")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nets = [Net(game, 4), Net(game, 4)]
    critics = [Net(game, 1) if algo != "reinforce" else None for _ in nets]
    opts = [torch.optim.Adam(
        [*n.parameters(), *(c.parameters() if c is not None else [])], lr=lr)
        for n, c in zip(nets, critics)]
    snaps = [[_frozen(n)] for n in nets]
    for it in range(iterations):
        if mode == "selfplay":
            batch = collect(game, _Current(nets[0]), _Current(nets[1]), episodes, horizon, rng)
            for i in (0, 1):
                update(nets[i], critics[i], opts, batch, i, algo, gamma, entropy)
        else:
            for i in (0, 1):
                mine = _Current(nets[i])
                theirs = _Mixture(snaps[1 - i], episodes, rng)
                players = (mine, theirs) if i == 0 else (theirs, mine)
                batch = collect(game, *players, episodes, horizon, rng)
                update(nets[i], critics[i], opts, batch, i, algo, gamma, entropy)
            if (it + 1) % snap_every == 0:
                for i in (0, 1):
                    snaps[i].append(_frozen(nets[i]))
    if mode == "selfplay":
        return Trained(_policy([nets[0]], horizon), _policy([nets[1]], horizon))
    return Trained(_policy(snaps[0], horizon), _policy(snaps[1], horizon))


def train_fp_br(
    game: SoccerGame, algo: str, rounds: int = 20, br_iters: int = 100, gamma: float = 0.9,
    horizon: int = 100, episodes: int = 64, lr: float = 1e-3, entropy: float = 0.01,
    seed: int = 0, on_round=None,
) -> Trained:
    """Fictitious play with best-response phases, all by policy gradient.

    Each round, each player runs ``br_iters`` policy-gradient iterations against
    the empirical average of the opponent's earlier best responses (a snapshot
    drawn uniformly per episode), then adds the resulting policy to its own
    history. The reported policy is the per-state average of a player's best
    responses, excluding the random initial network. ``on_round(r, trained)``
    is called after every round."""
    if algo not in ALGOS:
        raise ValueError("algo must be one of ALGOS")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nets = [Net(game, 4), Net(game, 4)]
    critics = [Net(game, 1) if algo != "reinforce" else None for _ in nets]
    opts = [torch.optim.Adam(
        [*n.parameters(), *(c.parameters() if c is not None else [])], lr=lr)
        for n, c in zip(nets, critics)]
    snaps = [[_frozen(n)] for n in nets]
    for r in range(1, rounds + 1):
        for i in (0, 1):
            for _ in range(br_iters):
                theirs = _Mixture(snaps[1 - i], episodes, rng)
                players = (_Current(nets[i]), theirs) if i == 0 else (theirs, _Current(nets[i]))
                batch = collect(game, *players, episodes, horizon, rng)
                update(nets[i], critics[i], opts, batch, i, algo, gamma, entropy)
            snaps[i].append(_frozen(nets[i]))
        if on_round is not None:
            on_round(r, Trained(_policy(snaps[0][1:], horizon), _policy(snaps[1][1:], horizon)))
    return Trained(_policy(snaps[0][1:], horizon), _policy(snaps[1][1:], horizon))
