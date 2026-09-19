"""Self-play REINFORCE for the zero-sum soccer Markov game -- the "does
policy gradient converge to a Nash equilibrium" question from the Sept 17
meeting, answered empirically rather than assumed.

Two independent policy networks, one per player, each mapping a state to a
softmax over the 4 actions -- **PyTorch**, per that meeting's own instruction
("use packages just for network training", the same point later repeated as
"PyTorch Neural Network... minibatch"). This is unlike
``soccer_nash/nash_dqn.py``'s hand-rolled NumPy net (predating that
instruction, kept as-is rather than rewritten under this change).

Training is on-policy self-play, matching the meeting's own derivation
exactly: the game used here is ``scoring="rate"`` (a goal scores and play
continues from a restart -- "this is an infinitely repeating game"), so
there is no natural episode boundary. Each training iteration rolls out one
long trajectory, then reuses *every suffix* of that single trajectory as its
own ``(state, return-from-here)`` pair --

    (s_0, R_1 + gamma R_2 + gamma^2 R_3 + ...)
    (s_1, R_2 + gamma R_3 + ...)
    (s_2, R_3 + ...)
    ...

-- exactly the reuse trick worked out on the whiteboard: a trajectory of
length T gives T training pairs from one rollout, not one. Both players'
losses are the plain REINFORCE gradient ``-E[log pi(a|s) * G]``; the
meeting's own derivation has no baseline/critic term, so neither does this
-- a baseline is the natural next step if variance turns out to be the
bottleneck, not assumed necessary in advance.

After training, both policies are checked against the *exact* solver's Q
matrix at every state -- never against each other -- via
:func:`soccer_nash.numerics.epsilon_equilibrium` (the "pi1 Q = Q^T pi2"
check from the meeting) and :func:`soccer_nash.exploit.duality_gap`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from soccer_nash.game import MOVE_ACTIONS, SoccerGame, State

_SCALE = torch.tensor([1 / 6, 1 / 4, 1 / 6, 1 / 4, 1.0])


class PolicyNet(nn.Module):
    """5 -> hidden -> hidden -> 4 logits, ReLU, softmax at the point of use."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(5, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x * _SCALE)

    def policy(self, state: State) -> np.ndarray:
        """The state's action distribution as a plain length-4 numpy array."""
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32)
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1).numpy()


@dataclass
class ReinforceResult:
    net0: PolicyNet
    net1: PolicyNet
    mean_reward_trace: list[float] = field(default_factory=list)


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Every suffix's discounted return, computed once by a backward pass --
    the trajectory-reuse trick: a length-T trajectory yields T pairs."""
    out = [0.0] * len(rewards)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def train_reinforce_selfplay(
    game: SoccerGame,
    gamma: float = 0.9,
    hidden: int = 64,
    iterations: int = 2000,
    rollout_len: int = 100,
    lr: float = 1e-3,
    seed: int = 0,
) -> ReinforceResult:
    """Self-play REINFORCE: both players act simultaneously every step, from
    their own policy network, on the real (stochastic) transition -- sampled
    via ``game.step``, not the exact expectation ``nash_dqn.py`` uses,
    because this is genuine on-policy reinforcement learning, not fitted-Q."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net0, net1 = PolicyNet(hidden), PolicyNet(hidden)
    opt0 = torch.optim.Adam(net0.parameters(), lr=lr)
    opt1 = torch.optim.Adam(net1.parameters(), lr=lr)

    trace: list[float] = []
    state = game.initial_state()
    for _ in range(iterations):
        states: list[State] = []
        a0s: list[int] = []
        a1s: list[int] = []
        r0s: list[float] = []

        for _t in range(rollout_len):
            x = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                d0 = torch.distributions.Categorical(logits=net0(x))
                d1 = torch.distributions.Categorical(logits=net1(x))
            a0, a1 = int(d0.sample()), int(d1.sample())
            ns, (r0, _r1), done = game.step(
                state, MOVE_ACTIONS[a0], MOVE_ACTIONS[a1], rng=rng
            )
            states.append(state)
            a0s.append(a0)
            a1s.append(a1)
            r0s.append(r0)
            state = game.initial_state() if done else ns

        g0 = _discounted_returns(r0s, gamma)
        g1 = [-g for g in g0]  # zero-sum: r1 == -r0 at every step, always

        X = torch.tensor(np.array(states), dtype=torch.float32)
        A0 = torch.tensor(a0s, dtype=torch.long)
        A1 = torch.tensor(a1s, dtype=torch.long)
        G0 = torch.tensor(g0, dtype=torch.float32)
        G1 = torch.tensor(g1, dtype=torch.float32)

        logp0 = torch.log_softmax(net0(X), dim=-1).gather(1, A0[:, None]).squeeze(1)
        loss0 = -(logp0 * G0).mean()
        opt0.zero_grad()
        loss0.backward()
        opt0.step()

        logp1 = torch.log_softmax(net1(X), dim=-1).gather(1, A1[:, None]).squeeze(1)
        loss1 = -(logp1 * G1).mean()
        opt1.zero_grad()
        loss1.backward()
        opt1.step()

        trace.append(float(np.mean(r0s)))

    return ReinforceResult(net0=net0, net1=net1, mean_reward_trace=trace)


def evaluate_policy_gradient(
    game: SoccerGame,
    result: ReinforceResult,
    exact_row_policy: dict[State, np.ndarray],
    exact_col_policy: dict[State, np.ndarray],
    exact_matrix_of,
    gamma: float = 0.9,
) -> dict[str, float]:
    """Score both self-play policies against the *exact* solver: action
    agreement, the per-state "pi1 Q = Q^T pi2" indifference check
    (``epsilon_equilibrium``, mean and max over all states), the whole-game
    exploitability (``duality_gap``) -- never compared to each other, always
    to the ground truth -- and each net's own expected discounted goal
    difference from kickoff against two fixed opponents: a **uniform-random**
    policy (``vs_random``) and the **best response to it** (``vs_best_response``,
    the exact worst case -- the opponent that has seen the net's mixing
    probabilities and replies optimally, not a random draw from them). A net
    that is close to the exact equilibrium should score similarly to
    ``exact`` against both; a net that merely names the right action often
    can still collapse against ``vs_best_response`` (see
    ``docs/tournament.md``'s "every deterministic offense has a perfect
    defense")."""
    from soccer_nash.evaluate import policy_value
    from soccer_nash.exploit import best_response_to, duality_gap, uniform_policy
    from soccer_nash.numerics import epsilon_equilibrium

    states = list(game.states())
    row_pol: dict[State, np.ndarray] = {}
    col_pol: dict[State, np.ndarray] = {}
    agree0 = agree1 = 0
    max_regret = 0.0
    mean_regret = 0.0
    for s in states:
        p, q = result.net0.policy(s), result.net1.policy(s)
        row_pol[s], col_pol[s] = p, q
        if np.argmax(p) == int(np.argmax(exact_row_policy[s])):
            agree0 += 1
        if np.argmax(q) == int(np.argmax(exact_col_policy[s])):
            agree1 += 1
        eps = epsilon_equilibrium(exact_matrix_of(s), p, q)
        max_regret = max(max_regret, eps)
        mean_regret += eps
    n = len(states)
    gap = duality_gap(game, row_pol, col_pol, gamma=gamma)

    s0 = game.initial_state()
    uniform = uniform_policy(game)
    row_vs_random = policy_value(game, row_pol, uniform, gamma=gamma)[s0]
    row_vs_br = best_response_to(game, row_pol, responder=1, gamma=gamma).values[s0]
    row_vs_br = -row_vs_br  # best_response_to returns the *responder's* (player 1's) value
    col_vs_random_vals = policy_value(game, uniform, col_pol, gamma=gamma)
    col_vs_random = -col_vs_random_vals[s0]  # player 1's own return, not player 0's
    col_vs_br = -best_response_to(game, col_pol, responder=0, gamma=gamma).values[s0]

    return {
        "row_action_agreement": agree0 / n,
        "col_action_agreement": agree1 / n,
        "mean_equilibrium_regret": mean_regret / n,
        "max_equilibrium_regret": max_regret,
        "duality_gap": gap,
        "row_vs_random": row_vs_random,
        "row_vs_best_response": row_vs_br,
        "col_vs_random": col_vs_random,
        "col_vs_best_response": col_vs_br,
    }
