"""Rock-paper-scissors with neural-network players: a small test of fictitious play before soccer.

Each player is a small network with a softmax output, trained by REINFORCE on sampled games.
``standard``: both current networks play each other. ``fictitious``: each trains against a
snapshot of the other drawn uniformly from its past snapshots; the reported (aggregate) policy
is the average of a player's snapshots. ``fictitious_argmax``: the same, but each snapshot first
becomes the pure policy that plays its most likely move, so the aggregate is the frequency with
which the snapshots play each move.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# payoff to the row player: rock, paper, scissors
PAYOFF = torch.tensor([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
MODES = ("standard", "fictitious", "fictitious_argmax")


def make_net() -> nn.Module:
    return nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 3))


def probs(net: nn.Module, pure: bool = False) -> torch.Tensor:
    logits = net(torch.ones(1, 1))[0]
    if pure:
        return nn.functional.one_hot(logits.argmax(), 3).float()
    return torch.softmax(logits, dim=-1)


def aggregate(snaps: list[nn.Module], pure: bool) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack([probs(n, pure) for n in snaps]).mean(0)


def exploitability(p_row: torch.Tensor, p_col: torch.Tensor) -> float:
    """What a best response gains against each side's policy, summed (0 at the equilibrium)."""
    return float((PAYOFF @ p_col).max() + (-(p_row @ PAYOFF)).max())


def run(mode: str, iterations: int = 600, batch: int = 256, lr: float = 0.03,
        snap_every: int = 5, seed: int = 0, entropy: float = 0.0, sgd: bool = False) -> dict:
    """Per iteration: the share of rock in the current network and in the aggregate policy
    (the average of snapshots for the fictitious modes, the current network for ``standard``)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    torch.manual_seed(seed)
    pure = mode == "fictitious_argmax"
    nets = [make_net(), make_net()]
    opt = torch.optim.SGD if sgd else torch.optim.Adam
    opts = [opt(n.parameters(), lr=lr) for n in nets]
    snaps = [[_copy(n)] for n in nets]
    out = {"current": [], "aggregate": [], "exploitability": []}
    for it in range(iterations):
        for i in (0, 1):
            theirs_snaps = snaps[1 - i]
            logits = nets[i](torch.ones(1, 1))[0]
            mine = torch.distributions.Categorical(logits=logits)
            a = mine.sample((batch,))
            if mode == "standard":
                with torch.no_grad():
                    b = torch.distributions.Categorical(probs=probs(nets[1 - i])).sample((batch,))
            else:
                pick = np.random.default_rng(seed * 100003 + it * 2 + i).integers(
                    len(theirs_snaps), size=batch)
                with torch.no_grad():
                    table = torch.stack([probs(n, pure) for n in theirs_snaps])
                    b = torch.distributions.Categorical(
                        probs=table[torch.from_numpy(pick)]).sample()
            payoff = PAYOFF if i == 0 else -PAYOFF.T
            reward = payoff[a, b]
            loss = -(mine.log_prob(a) * (reward - reward.mean())).mean() - entropy * mine.entropy()
            opts[i].zero_grad()
            loss.backward()
            opts[i].step()
        if mode != "standard" and (it + 1) % snap_every == 0:
            for i in (0, 1):
                snaps[i].append(_copy(nets[i]))
        with torch.no_grad():
            cur = [probs(n) for n in nets]
        agg = cur if mode == "standard" else [aggregate(snaps[i], pure) for i in (0, 1)]
        out["current"].append(round(float(cur[0][0]), 5))
        out["aggregate"].append(round(float(agg[0][0]), 5))
        out["exploitability"].append(round(exploitability(agg[0], agg[1]), 5))
    return out


def _copy(net: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(net).requires_grad_(False)
