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
EXACT_ALGOS = ("a2c_exact", "a2c_pretrained")
MODES = ("selfplay", "fictitious", "fictitious_argmax")


def features(states: list[State], t: int, horizon: int) -> torch.Tensor:
    """State plus the fraction of the horizon still remaining."""
    x = torch.tensor(np.array(states, dtype=np.float32))
    return torch.cat([x, torch.full((len(states), 1), (horizon - t) / horizon)], dim=1)


def fit_exact_critic(critic, exact, player: int, epochs: int = 10,
                     batch_size: int = 4096, lr: float = 1e-3) -> None:
    """Fit a value network to all exact state-time values, then freeze it."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("critic pretraining epochs and batch size must be positive")
    states = list(exact.states)
    x = torch.cat([features(states, t, exact.horizon) for t in range(exact.horizon)])
    ids = torch.tensor([exact.code(*s) for _ in range(exact.horizon) for s in states])
    times = torch.arange(exact.horizon).repeat_interleave(len(states))
    sign = 1.0 if player == 0 else -1.0
    y = exact.V[times, ids] * sign
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    critic.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(x)).split(batch_size):
            loss = nn.functional.mse_loss(critic(x[idx]).squeeze(-1), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    critic.eval().requires_grad_(False)


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


class Head(nn.Module):
    """One player's slice of a network shared by both players: the hidden layers are
    common, the output layer holds 4 logits per player."""

    def __init__(self, base: Net, player: int):
        super().__init__()
        self.base, self.player = base, player

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)[..., 4 * self.player:4 * self.player + 4]


def _frozen(net: Net) -> Net:
    return copy.deepcopy(net).requires_grad_(False)


class _Current:
    def __init__(self, net: Net):
        self.net = net

    def probs(self, x: torch.Tensor, ep: np.ndarray) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


def _probs(net: Net, x: torch.Tensor, pure: bool) -> torch.Tensor:
    """The softmax policy; with ``pure``, the deterministic policy on its most likely move."""
    logits = net(x)
    if pure:
        return nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return torch.softmax(logits, dim=-1)


class _Mixture:
    """Each episode plays one snapshot, drawn uniformly -- the empirical average. With ``pure``
    every snapshot plays its most likely move, so the mixture is an average of pure policies."""
    def __init__(self, snaps: list[Net], n_episodes: int, rng: np.random.Generator,
                 pure: bool = False):
        self.snaps = snaps
        self.pure = pure
        self.pick = rng.integers(len(snaps), size=n_episodes)

    def probs(self, x: torch.Tensor, ep: np.ndarray) -> torch.Tensor:
        out = torch.zeros(len(ep), 4)
        which = self.pick[ep]
        for k in np.unique(which):
            m = torch.from_numpy(which == k)
            out[m] = _probs(self.snaps[int(k)], x[m], self.pure)
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


def update(
    net, critic, opts, batch, player, algo, gamma, entropy, clip=0.2, epochs=4, exact=None,
    trim=0, minibatch=0,
):
    sign = 1.0 if player == 0 else -1.0
    X, alive = batch["X"], batch["alive"]
    r = sign * batch["R"]
    if algo == "a2c_exact":
        value, target = None, None
        adv = sign * exact.advantage(batch)
    elif algo == "a2c_pretrained":
        with torch.no_grad():
            value = critic(X).squeeze(-1)
            adv = sign * exact.action_value(batch) - value
        target = None
    else:
        with torch.no_grad():
            value = critic(X).squeeze(-1) if critic is not None else None
        adv, target = _returns(r, alive, value, algo, gamma)
    use = alive.clone()
    if trim:
        use[X.shape[0] - trim:] = False   # earlier returns still use them; the loss does not
    x, a = X[use], batch["A"][player][use]
    old, adv_f = batch["LP"][player][use], adv[use]
    if algo == "ppo":
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
    elif algo == "a2c":
        adv_f = adv_f - value[use]
    tgt = target[use] if target is not None else None
    for _ in range(epochs if algo == "ppo" else 1):
        # one gradient step on all the data, or one per random mini-batch of ``minibatch`` samples
        parts = torch.randperm(x.shape[0]).split(minibatch) if minibatch else [slice(None)]
        for i in parts:
            logp_all = torch.log_softmax(net(x[i]), dim=-1)
            logp = logp_all.gather(1, a[i][:, None]).squeeze(1)
            if algo == "ppo":
                ratio = (logp - old[i]).exp()
                pl = -torch.min(ratio * adv_f[i], ratio.clamp(1 - clip, 1 + clip) * adv_f[i]).mean()
            else:
                pl = -(logp * adv_f[i]).mean()
            ent = -(logp_all.exp() * logp_all).sum(-1).mean()
            loss = pl - entropy * ent
            if tgt is not None:
                loss = loss + 0.5 * nn.functional.mse_loss(critic(x[i]).squeeze(-1), tgt[i])
            opts[player].zero_grad()
            loss.backward()
            opts[player].step()


def _concat(batches: list[dict]) -> dict:
    """Several (T, E) batches side by side along the episode axis."""
    if len(batches) == 1:
        return batches[0]
    return {"X": torch.cat([b["X"] for b in batches], 1),
            "A": torch.cat([b["A"] for b in batches], 2),
            "LP": torch.cat([b["LP"] for b in batches], 2),
            "R": torch.cat([b["R"] for b in batches], 1),
            "alive": torch.cat([b["alive"] for b in batches], 1)}


@dataclass
class Trained:
    """Reported policies: ``pol0`` / ``pol1(t, states) -> (n, 4)``."""
    pol0: object
    pol1: object
    snaps: tuple | None = None   # each player's snapshots, for the fictitious-play modes
    critics: tuple | None = None # frozen value networks when using a2c_pretrained


def _policy(nets: list[Net], horizon: int, pure: bool = False):
    def pol(t, states):
        with torch.no_grad():
            x = features(list(states), t, horizon)
            return torch.stack([_probs(n, x, pure) for n in nets]).mean(0).numpy()
    return pol


def weight_average(snaps: list[Net], horizon: int, last_layer_only: bool = False):
    """The policy of a network whose weights are the mean of the snapshots' weights. With
    ``last_layer_only`` the hidden layers are the last snapshot's and only the output layer is
    averaged. Not the fictitious-play average (that averages policies); kept as a comparison."""
    def linears(net):
        return [m for m in net.body if isinstance(m, nn.Linear)]

    avg = copy.deepcopy(snaps[-1]).requires_grad_(False)
    with torch.no_grad():
        for k, layer in enumerate(linears(avg)):
            if last_layer_only and k < len(linears(avg)) - 1:
                continue
            for name in ("weight", "bias"):
                mean = torch.stack([getattr(linears(sn)[k], name) for sn in snaps]).mean(0)
                getattr(layer, name).copy_(mean)
    return _policy([avg], horizon)


def train(
    game: SoccerGame, algo: str, mode: str = "selfplay", gamma: float = 0.9,
    horizon: int = 100, iterations: int = 300, episodes: int = 64, lr: float = 1e-3,
    entropy: float = 0.01, snap_every: int = 5, seed: int = 0, exact=None,
    shared: bool = False, trim: int = 0, replay: int = 1, minibatch: int = 0,
    checkpoint_every: int = 0, on_checkpoint=None, critic_pretrain_epochs: int = 10,
    critic_pretrain_batch_size: int = 4096,
) -> Trained:
    """``mode``: ``selfplay`` (standard), ``fictitious`` (train against the average of the
    opponent's past snapshots, report the average of one's own), or ``fictitious_argmax``
    (the same, but every snapshot is first turned into the pure policy that takes its most
    likely move, so both the opponent mixture and the reported policy average pure policies).
    ``replay``: each update uses the last ``replay`` iterations' data; ``minibatch``: split the
    data into random mini-batches of that many samples, one gradient step each.
    ``on_checkpoint(iteration, reported, current)`` is called every ``checkpoint_every``
    iterations with the reported policies and the current networks' policies.
    ``shared``: both players' policies are two output heads of one network.
    ``trim``: drop the last ``trim`` steps of every episode from the policy and critic
    losses (they still enter the returns of earlier steps)."""
    if (algo not in ALGOS and algo not in EXACT_ALGOS) or mode not in MODES:
        raise ValueError(f"algo: one of ALGOS or {EXACT_ALGOS}; mode: one of {MODES}")
    if algo in EXACT_ALGOS and exact is None:
        raise ValueError(f"{algo} needs the exact solver values")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if shared:
        base = Net(game, 8)
        nets = [Head(base, 0), Head(base, 1)]
    else:
        nets = [Net(game, 4), Net(game, 4)]
    critics = [Net(game, 1) if algo not in ("reinforce", "a2c_exact") else None for _ in nets]
    if algo == "a2c_pretrained":
        for i, critic in enumerate(critics):
            fit_exact_critic(critic, exact, i, critic_pretrain_epochs,
                             critic_pretrain_batch_size, lr)
    if shared:
        params = [*base.parameters(),
                  *(q for c in critics if c is not None for q in c.parameters())]
        opts = [torch.optim.Adam(params, lr=lr)] * 2
    else:
        opts = [torch.optim.Adam(
            [*n.parameters(), *(c.parameters() if c is not None else [])], lr=lr)
            for n, c in zip(nets, critics)]
    snaps = [[_frozen(n)] for n in nets]
    pure = mode == "fictitious_argmax"
    buffers = [[], []]   # the last ``replay`` batches of each player
    for it in range(iterations):
        if mode == "selfplay":
            batch = collect(game, _Current(nets[0]), _Current(nets[1]), episodes, horizon, rng)
            for i in (0, 1):
                buffers[i] = (buffers[i] + [batch])[-replay:]
                update(nets[i], critics[i], opts, _concat(buffers[i]), i, algo, gamma, entropy,
                       exact=exact, trim=trim, minibatch=minibatch)
        else:
            for i in (0, 1):
                mine = _Current(nets[i])
                theirs = _Mixture(snaps[1 - i], episodes, rng, pure)
                players = (mine, theirs) if i == 0 else (theirs, mine)
                batch = collect(game, *players, episodes, horizon, rng)
                buffers[i] = (buffers[i] + [batch])[-replay:]
                update(nets[i], critics[i], opts, _concat(buffers[i]), i, algo, gamma, entropy,
                       exact=exact, trim=trim, minibatch=minibatch)
            if (it + 1) % snap_every == 0:
                for i in (0, 1):
                    snaps[i].append(_frozen(nets[i]))
        if checkpoint_every and on_checkpoint and (it + 1) % checkpoint_every == 0:
            reported = (Trained(_policy([nets[0]], horizon), _policy([nets[1]], horizon))
                        if mode == "selfplay" else
                        Trained(_policy(snaps[0], horizon, pure), _policy(snaps[1], horizon, pure)))
            on_checkpoint(it + 1, reported,
                          Trained(_policy([nets[0]], horizon), _policy([nets[1]], horizon)))
    if mode == "selfplay":
        return Trained(_policy([nets[0]], horizon), _policy([nets[1]], horizon),
                       critics=tuple(critics) if algo == "a2c_pretrained" else None)
    return Trained(_policy(snaps[0], horizon, pure), _policy(snaps[1], horizon, pure),
                   (snaps[0], snaps[1]),
                   tuple(critics) if algo == "a2c_pretrained" else None)


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
