"""A provisional continuous dog-and-sheep game with an angle-radius policy.

Nothing about this game has been specified yet; every choice below is a
placeholder (see ``docs/solver_assumptions.md``, S14-S16, and the questions
for the professor). A dog and a sheep move in the unit square. Each step both
pick an angle ``t`` and a radius ``r <= speed`` and step by ``r (cos t, sin t)``.
The dog wins (+1) by getting within ``CAPTURE`` of the sheep, and loses (-1) if
the sheep lasts ``HORIZON`` steps. Zero-sum.

The policy head follows the note: a network outputs an angle in ``[0, 2 pi)``
(von Mises) and a radius in ``[0, speed]`` (scaled Beta), sampled independently.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, VonMises

DOG_SPEED, SHEEP_SPEED = 0.06, 0.04
CAPTURE = 0.05
HORIZON = 100
DOG_START, SHEEP_START = (0.2, 0.2), (0.8, 0.8)


def features(dog: torch.Tensor, sheep: torch.Tensor, t: int) -> torch.Tensor:
    rel = sheep - dog
    time = torch.full((dog.shape[0], 1), t / HORIZON)
    return torch.cat([dog, sheep, rel, time], dim=1)


def move(pos: torch.Tensor, theta: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    step = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    return (pos + step).clamp(0.0, 1.0)


class PolarPolicy(nn.Module):
    def __init__(self, speed: float, hidden: int = 64):
        super().__init__()
        self.speed = speed
        self.body = nn.Sequential(
            nn.Linear(7, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 5),
        )

    def dists(self, x: torch.Tensor) -> tuple[VonMises, Beta]:
        ux, uy, kappa, a, b = self.body(x).unbind(-1)
        angle = VonMises(torch.atan2(uy, ux), nn.functional.softplus(kappa) + 0.1)
        radius = Beta(nn.functional.softplus(a) + 1.0, nn.functional.softplus(b) + 1.0)
        return angle, radius

    def act(self, x: torch.Tensor, greedy: bool = False):
        angle, radius = self.dists(x)
        if greedy:
            theta, frac = angle.loc, radius.mean
        else:
            theta, frac = angle.sample(), radius.sample().clamp(1e-4, 1 - 1e-4)
        return theta, frac

    def log_prob(self, x, theta, frac) -> torch.Tensor:
        angle, radius = self.dists(x)
        return angle.log_prob(theta) + radius.log_prob(frac)


def _value_net(hidden: int = 64) -> nn.Module:
    return nn.Sequential(nn.Linear(7, hidden), nn.Tanh(), nn.Linear(hidden, 1))


def play(dog_act, sheep_act, n_games: int = 200, seed: int = 0) -> dict[str, float]:
    """``dog_act(x) -> (theta, frac)`` and likewise for the sheep. Returns the
    dog's capture rate and the mean capture time (steps) over captured games."""
    torch.manual_seed(seed)
    dog = torch.tensor([DOG_START] * n_games)
    sheep = torch.tensor([SHEEP_START] * n_games)
    caught = torch.zeros(n_games, dtype=torch.bool)
    when = torch.zeros(n_games)
    for t in range(HORIZON):
        x = features(dog, sheep, t)
        td, fd = dog_act(x)
        ts, fs = sheep_act(x)
        dog = move(dog, td, fd * DOG_SPEED)
        sheep = move(sheep, ts, fs * SHEEP_SPEED)
        hit = (torch.linalg.norm(dog - sheep, dim=1) <= CAPTURE) & ~caught
        when[hit] = t + 1
        caught |= hit
    return {
        "capture_rate": float(caught.float().mean()),
        "mean_capture_step": float(when[caught].mean()) if caught.any() else float("nan"),
    }


def random_act(x: torch.Tensor):
    n = x.shape[0]
    return torch.rand(n) * 2 * np.pi, torch.rand(n)


def greedy_dog_act(x: torch.Tensor):
    rel = x[:, 4:6]
    return torch.atan2(rel[:, 1], rel[:, 0]), torch.ones(x.shape[0])


def train_ppo_selfplay(
    iterations: int = 200, n_envs: int = 64, gamma: float = 0.99, lam: float = 0.95,
    clip: float = 0.2, epochs: int = 4, lr: float = 3e-4, seed: int = 0,
) -> tuple[PolarPolicy, PolarPolicy]:
    """Self-play PPO for both players on sparse terminal rewards."""
    torch.manual_seed(seed)
    pols = [PolarPolicy(DOG_SPEED), PolarPolicy(SHEEP_SPEED)]
    vals = [_value_net(), _value_net()]
    opts = [torch.optim.Adam([*p.parameters(), *v.parameters()], lr=lr)
            for p, v in zip(pols, vals)]
    for _ in range(iterations):
        dog = torch.tensor([DOG_START] * n_envs)
        sheep = torch.tensor([SHEEP_START] * n_envs)
        alive = torch.ones(n_envs, dtype=torch.bool)
        buf = {k: [] for k in ("x", "th", "fr", "lp", "alive", "rew")}
        buf = [dict(buf), dict(buf)]
        for t in range(HORIZON):
            x = features(dog, sheep, t)
            with torch.no_grad():
                acts = [p.act(x) for p in pols]
                lps = [p.log_prob(x, *a) for p, a in zip(pols, acts)]
            dog = move(dog, acts[0][0], acts[0][1] * DOG_SPEED)
            sheep = move(sheep, acts[1][0], acts[1][1] * SHEEP_SPEED)
            hit = torch.linalg.norm(dog - sheep, dim=1) <= CAPTURE
            r_dog = torch.zeros(n_envs)
            r_dog[hit & alive] = 1.0
            if t == HORIZON - 1:
                r_dog[~hit & alive] = -1.0
            for i, sign in ((0, 1.0), (1, -1.0)):
                b = buf[i]
                b["x"].append(x)
                b["th"].append(acts[i][0])
                b["fr"].append(acts[i][1])
                b["lp"].append(lps[i])
                b["alive"].append(alive.clone())
                b["rew"].append(sign * r_dog)
            alive = alive & ~hit
        for i in (0, 1):
            _ppo_update(pols[i], vals[i], opts[i], buf[i], gamma, lam, clip, epochs)
    return pols[0], pols[1]


def _ppo_update(policy, value, opt, b, gamma, lam, clip, epochs) -> None:
    X = torch.stack(b["x"])                    # (T, E, 7)
    alive = torch.stack(b["alive"])
    rew = torch.stack(b["rew"])
    T = X.shape[0]
    with torch.no_grad():
        v = value(X).squeeze(-1)
    adv = torch.zeros_like(rew)
    g = torch.zeros(rew.shape[1])
    for t in reversed(range(T)):
        # an env's episode ends after a capture (alive[t+1] false) or at the horizon
        cont = alive[t + 1].float() if t + 1 < T else torch.zeros(rew.shape[1])
        nxt = v[t + 1] if t + 1 < T else torch.zeros(rew.shape[1])
        delta = rew[t] + gamma * cont * nxt - v[t]
        g = delta + gamma * lam * cont * g
        adv[t] = g
    ret = adv + v
    mask = alive.flatten()
    flat = lambda a: a.flatten(0, 1)[mask]  # noqa: E731
    x, th, fr = flat(X), flat(torch.stack(b["th"])), flat(torch.stack(b["fr"]))
    old_lp, a_, r_ = flat(torch.stack(b["lp"])), flat(adv), flat(ret)
    a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
    for _ in range(epochs):
        ratio = (policy.log_prob(x, th, fr) - old_lp).exp()
        pl = -torch.min(ratio * a_, ratio.clamp(1 - clip, 1 + clip) * a_).mean()
        vl = nn.functional.mse_loss(value(x).squeeze(-1), r_)
        opt.zero_grad()
        (pl + 0.5 * vl).backward()
        opt.step()
