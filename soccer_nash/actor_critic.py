"""Self-play A2C and PPO on the soccer Markov game -- the "REINFORCE / A2C /
PPO to replicate the exact solver" item from the research-group note.

Same setup as :func:`soccer_nash.policy_gradient.train_reinforce_selfplay`
(two independent softmax policy nets, simultaneous play, real stochastic
``game.step`` transitions, per-player critic) so the three algorithms are
directly comparable and all score through the same
:func:`~soccer_nash.policy_gradient.evaluate_policy_gradient` against the exact
solve. What differs is only the advantage estimate and the policy update:

* **A2C** -- n-step bootstrapped return ``G_t = r_t + g(1-done_t) G_{t+1}``,
  ``G_T = V(s_T)`` (the critic bootstraps; REINFORCE's baseline does not).
  One gradient step per rollout batch.
* **PPO** -- GAE(lambda) advantages, clipped surrogate objective, several
  epochs of full-batch updates per rollout batch.
"""

from __future__ import annotations

import numpy as np
import torch

from soccer_nash.game import MOVE_ACTIONS, SoccerGame
from soccer_nash.policy_gradient import PolicyNet, ReinforceResult, ValueNet


def _collect(game, net0, net1, v0, v1, state, n_steps, rng):
    """One rollout of ``n_steps``; returns tensors plus the bootstrap values."""
    S, A0, A1, R, D, LP0, LP1 = [], [], [], [], [], [], []
    for _ in range(n_steps):
        x = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            d0 = torch.distributions.Categorical(logits=net0(x))
            d1 = torch.distributions.Categorical(logits=net1(x))
            a0, a1 = d0.sample(), d1.sample()
            LP0.append(float(d0.log_prob(a0)))
            LP1.append(float(d1.log_prob(a1)))
        ns, (r0, _), done = game.step(
            state, MOVE_ACTIONS[int(a0)], MOVE_ACTIONS[int(a1)], rng=rng)
        S.append(state)
        A0.append(int(a0))
        A1.append(int(a1))
        R.append(r0)
        D.append(done)
        state = game.initial_state() if done else ns
    X = torch.tensor(np.array(S), dtype=torch.float32)
    xe = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        val0, val1 = v0(X), v1(X)
        last0, last1 = float(v0(xe)), float(v1(xe))
    return {
        "X": X, "A0": torch.tensor(A0), "A1": torch.tensor(A1),
        "R": np.array(R, dtype=np.float64), "D": np.array(D, dtype=bool),
        "LP0": torch.tensor(LP0, dtype=torch.float32),
        "LP1": torch.tensor(LP1, dtype=torch.float32),
        "val0": val0.numpy().astype(np.float64),
        "val1": val1.numpy().astype(np.float64),
        "last0": last0, "last1": last1, "end_state": state,
    }


def _advantages(r, d, val, last, gamma, lam):
    """GAE(lambda); ``lam=1`` is the n-step bootstrapped advantage (A2C).
    Returns (advantage, return-target = advantage + value)."""
    T = len(r)
    adv = np.zeros(T)
    g = 0.0
    for t in reversed(range(T)):
        nxt = last if t == T - 1 else val[t + 1]
        nonterm = 0.0 if d[t] else 1.0
        delta = r[t] + gamma * nonterm * nxt - val[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    return adv, adv + val


def train_actor_critic_selfplay(
    game: SoccerGame,
    algo: str = "a2c",
    gamma: float = 0.9,
    hidden: int = 64,
    iterations: int = 2000,
    rollout_len: int = 100,
    lr: float = 1e-3,
    seed: int = 0,
    entropy_coef: float = 0.01,
    gae_lambda: float = 0.95,
    clip: float = 0.2,
    ppo_epochs: int = 4,
    init_net0: PolicyNet | None = None,
    init_net1: PolicyNet | None = None,
) -> ReinforceResult:
    if algo not in ("a2c", "ppo"):
        raise ValueError("algo must be 'a2c' or 'ppo'")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nets = [PolicyNet(hidden), PolicyNet(hidden)]
    for n, init in zip(nets, (init_net0, init_net1)):
        if init is not None:
            n.load_state_dict(init.state_dict())
    vals = [ValueNet(hidden), ValueNet(hidden)]
    popt = [torch.optim.Adam(n.parameters(), lr=lr) for n in nets]
    vopt = [torch.optim.Adam(v.parameters(), lr=lr) for v in vals]
    lam = 1.0 if algo == "a2c" else gae_lambda
    epochs = 1 if algo == "a2c" else ppo_epochs

    trace: list[float] = []
    state = game.initial_state()
    for _ in range(iterations):
        b = _collect(game, nets[0], nets[1], vals[0], vals[1], state,
                     rollout_len, rng)
        state = b["end_state"]
        trace.append(float(b["R"].mean()))
        for i in (0, 1):
            sign = 1.0 if i == 0 else -1.0  # zero-sum: r1 = -r0
            r = sign * b["R"]
            val = b[f"val{i}"]
            adv, target = _advantages(r, b["D"], val, b[f"last{i}"], gamma, lam)
            adv_t = torch.tensor(adv, dtype=torch.float32)
            tgt_t = torch.tensor(target, dtype=torch.float32)
            if algo == "ppo":
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            acts = b[f"A{i}"]
            old_lp = b[f"LP{i}"]
            for _e in range(epochs):
                logits = nets[i](b["X"])
                logp_all = torch.log_softmax(logits, dim=-1)
                logp = logp_all.gather(1, acts[:, None]).squeeze(1)
                ent = -(logp_all.exp() * logp_all).sum(-1).mean()
                if algo == "a2c":
                    pl = -(logp * adv_t).mean()
                else:
                    ratio = (logp - old_lp).exp()
                    pl = -torch.min(
                        ratio * adv_t,
                        torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t,
                    ).mean()
                loss = pl - entropy_coef * ent
                popt[i].zero_grad()
                loss.backward()
                popt[i].step()
                vloss = torch.nn.functional.mse_loss(vals[i](b["X"]), tgt_t)
                vopt[i].zero_grad()
                vloss.backward()
                vopt[i].step()
    return ReinforceResult(net0=nets[0], net1=nets[1], mean_reward_trace=trace)
