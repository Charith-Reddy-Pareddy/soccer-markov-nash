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
from typing import Protocol

import numpy as np
import torch
from torch import nn

from soccer_nash.game import MOVE_ACTIONS, SoccerGame, State


class _PolicySource(Protocol):
    """Anything that maps a state to an action distribution -- a plain
    ``PolicyNet`` (:func:`train_reinforce_selfplay`) or a
    ``_MirrorPolicyView`` over a shared/partial-share net
    (:func:`train_reinforce_selfplay_shared`)."""

    def policy(self, state: State) -> np.ndarray: ...

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


class ValueNet(nn.Module):
    """5 -> hidden -> hidden -> 1: a state-value baseline for REINFORCE's
    variance, trained by regression onto the observed return ``G``. Not a
    critic in the TD sense -- there is no bootstrapping, just ``E[(V(s) -
    G)^2]``, the plain "reduce variance without changing the expected
    gradient" baseline."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(5, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x * _SCALE).squeeze(-1)


@dataclass
class ReinforceResult:
    net0: _PolicySource
    net1: _PolicySource
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


def _rollout(
    game: SoccerGame,
    net0: PolicyNet,
    net1: PolicyNet,
    start_state: State,
    rollout_len: int,
    rng: np.random.Generator,
) -> tuple[list[State], list[int], list[int], list[float], State]:
    """One on-policy self-play rollout of ``rollout_len`` steps from
    ``start_state``. Returns the per-step states/actions/rewards plus the
    state the rollout ended on (so a caller can continue from it)."""
    state = start_state
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
    return states, a0s, a1s, r0s, state


def pretrain_policy_nets(
    game: SoccerGame,
    exact_row_policy: dict[State, np.ndarray],
    exact_col_policy: dict[State, np.ndarray],
    hidden: int = 64,
    epochs: int = 400,
    lr: float = 3e-3,
    seed: int = 0,
) -> tuple[PolicyNet, PolicyNet]:
    """Regress ``PolicyNet``s directly onto the exact equilibrium strategies
    -- supervised cross-entropy against the target distribution, no rollouts,
    no self-play -- so the meeting's "would pre-training help?" question has
    an actual starting point to hand :func:`train_reinforce_selfplay`
    (``init_net0``/``init_net1``), the same "fit to exact, then continue
    training" shape as ``nash_dqn.fit_q_to_exact`` + ``train_nash_dqn``'s
    warm start, adapted to REINFORCE's on-policy loop instead of TD
    bootstrap."""
    torch.manual_seed(seed)
    states = list(game.states())
    X = torch.tensor(np.array(states), dtype=torch.float32)
    P0 = torch.tensor(np.array([exact_row_policy[s] for s in states]), dtype=torch.float32)
    P1 = torch.tensor(np.array([exact_col_policy[s] for s in states]), dtype=torch.float32)

    def _fit(net: PolicyNet, target: torch.Tensor) -> PolicyNet:
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _epoch in range(epochs):
            logp = torch.log_softmax(net(X), dim=-1)
            loss = -(target * logp).sum(dim=-1).mean()  # cross-entropy to a soft target
            opt.zero_grad()
            loss.backward()
            opt.step()
        return net

    net0 = _fit(PolicyNet(hidden), P0)
    net1 = _fit(PolicyNet(hidden), P1)
    return net0, net1


def train_reinforce_selfplay(
    game: SoccerGame,
    gamma: float = 0.9,
    hidden: int = 64,
    iterations: int = 2000,
    rollout_len: int = 100,
    n_rollouts: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
    init_net0: PolicyNet | None = None,
    init_net1: PolicyNet | None = None,
    use_baseline: bool = False,
    entropy_coef: float = 0.0,
) -> ReinforceResult:
    """Self-play REINFORCE: both players act simultaneously every step, from
    their own policy network, on the real (stochastic) transition -- sampled
    via ``game.step``, not the exact expectation ``nash_dqn.py`` uses,
    because this is genuine on-policy reinforcement learning, not fitted-Q.

    ``init_net0``/``init_net1``, when given, seed the starting weights (e.g.
    from :func:`pretrain_policy_nets`) instead of a random initialization --
    lets a caller compare "from scratch" against "warm-started" self-play,
    same shape as ``nash_dqn.train_nash_dqn``'s ``init_net``.

    ``n_rollouts`` (default 1, the original behaviour): with ``n_rollouts >
    1``, each iteration collects ``n_rollouts`` independent rollouts instead
    of one -- the *single* ongoing trajectory (rollout 0, carried over
    between iterations exactly as before) plus ``n_rollouts - 1`` fresh
    rollouts each restarted at ``game.initial_state()`` -- and trains on all
    of them batched together in one gradient step. The single-trajectory
    suffixes reused within each rollout are correlated by construction (each
    state depends on the last); stacking several *independent* rollouts into
    the same batch is the further variance-reduction step that on its own
    -- each rollout's discounted returns are computed separately so a
    boundary between two rollouts is never treated as a continuation of one
    trajectory.

    ``use_baseline`` and ``entropy_coef`` are two independent, separately
    toggleable levers, deliberately not bundled into one "improved REINFORCE"
    flag -- they target different failure modes and this repo's own findings
    (self-play collapsing to a state-independent policy under aggressive
    batching) call for telling them apart, not a single combined ablation:

    * ``use_baseline=True`` subtracts a learned state-value baseline
      (:class:`ValueNet`, one per player, trained by regression onto the
      observed return) from ``G`` before the policy loss -- ``advantage =
      G - V(s).detach()``. This is pure variance reduction: it does not
      change what the expected gradient points toward, only how noisy each
      sample estimate of it is.
    * ``entropy_coef > 0`` adds ``-entropy_coef * H(pi(.|s))`` to the policy
      loss, directly rewarding a less-peaked distribution -- a mechanism
      for preventing premature collapse to a near-deterministic policy, not
      a variance-reduction technique at all.

    Both default to off, so the original from-scratch/batched results are
    unchanged unless a caller opts in."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net0 = PolicyNet(hidden)
    net1 = PolicyNet(hidden)
    if init_net0 is not None:
        net0.load_state_dict(init_net0.state_dict())
    if init_net1 is not None:
        net1.load_state_dict(init_net1.state_dict())
    opt0 = torch.optim.Adam(net0.parameters(), lr=lr)
    opt1 = torch.optim.Adam(net1.parameters(), lr=lr)

    if use_baseline:
        value_net0 = ValueNet(hidden)
        value_net1 = ValueNet(hidden)
        vopt0 = torch.optim.Adam(value_net0.parameters(), lr=lr)
        vopt1 = torch.optim.Adam(value_net1.parameters(), lr=lr)

    trace: list[float] = []
    state = game.initial_state()
    for _ in range(iterations):
        states: list[State] = []
        a0s: list[int] = []
        a1s: list[int] = []
        g0s: list[float] = []
        raw_r0: list[float] = []

        for k in range(n_rollouts):
            start = state if k == 0 else game.initial_state()
            s_k, a0_k, a1_k, r0_k, end_state = _rollout(game, net0, net1, start, rollout_len, rng)
            if k == 0:
                state = end_state  # only the "main" trajectory carries over between iterations
            states += s_k
            a0s += a0_k
            a1s += a1_k
            g0s += _discounted_returns(r0_k, gamma)  # per-rollout: no bleed across boundaries
            raw_r0 += r0_k

        g1s = [-g for g in g0s]  # zero-sum: r1 == -r0 at every step, always

        X = torch.tensor(np.array(states), dtype=torch.float32)
        A0 = torch.tensor(a0s, dtype=torch.long)
        A1 = torch.tensor(a1s, dtype=torch.long)
        G0 = torch.tensor(g0s, dtype=torch.float32)
        G1 = torch.tensor(g1s, dtype=torch.float32)

        adv0, adv1 = G0, G1
        if use_baseline:
            v0 = value_net0(X)
            v1 = value_net1(X)
            adv0 = G0 - v0.detach()
            adv1 = G1 - v1.detach()
            vloss0 = torch.nn.functional.mse_loss(v0, G0)
            vopt0.zero_grad()
            vloss0.backward()
            vopt0.step()
            vloss1 = torch.nn.functional.mse_loss(v1, G1)
            vopt1.zero_grad()
            vloss1.backward()
            vopt1.step()

        logits0 = net0(X)
        dist0 = torch.distributions.Categorical(logits=logits0)
        logp0 = torch.log_softmax(logits0, dim=-1).gather(1, A0[:, None]).squeeze(1)
        loss0 = -(logp0 * adv0).mean() - entropy_coef * dist0.entropy().mean()
        opt0.zero_grad()
        loss0.backward()
        opt0.step()

        logits1 = net1(X)
        dist1 = torch.distributions.Categorical(logits=logits1)
        logp1 = torch.log_softmax(logits1, dim=-1).gather(1, A1[:, None]).squeeze(1)
        loss1 = -(logp1 * adv1).mean() - entropy_coef * dist1.entropy().mean()
        opt1.zero_grad()
        loss1.backward()
        opt1.step()

        trace.append(float(np.mean(raw_r0)))

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


class JointPolicyNet(nn.Module):
    """A single network reading the *raw* joint state directly and
    outputting both players' action logits (4 for player 0, 4 for player 1)
    from one shared body all the way to the output layer -- the "fully
    shared" architecture. No symmetry transform, no mirroring: both
    policies are read off the same forward pass on the same input, and
    whatever relationship training finds between them, symmetric or not,
    is learned rather than imposed (see :func:`train_reinforce_selfplay_shared`
    for why that distinction matters)."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(5, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x * _SCALE)

    def logits0(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[..., :4]

    def logits1(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[..., 4:]


class SharedTrunkPolicyNet(nn.Module):
    """A shared body (geometry feature extraction, identical weights for
    both players) with a separate linear head per player -- the "partial
    sharing" architecture: the two players' final decision layer can
    diverge, but the layers that turn raw coordinates into features cannot.
    Both players read the same raw state; no symmetry transform."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(5, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head0 = nn.Linear(hidden, 4)
        self.head1 = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor, player: int) -> torch.Tensor:
        z = self.body(x * _SCALE)
        return self.head0(z) if player == 0 else self.head1(z)


class _HeadPolicyView:
    """Adapter so :func:`evaluate_policy_gradient`'s ``result.net0.policy(s)``
    / ``result.net1.policy(s)`` interface works for the shared/partial
    architectures too -- reads the given (possibly shared) network directly
    on the raw state. Deliberately this simple: not imposing any structure
    on the relationship between the two players' policies is the entire
    point of this architecture (see :func:`train_reinforce_selfplay_shared`)."""

    def __init__(self, logits_fn):
        self._logits_fn = logits_fn

    def policy(self, state: State) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32)
            return torch.softmax(self._logits_fn(x), dim=-1).numpy()


def _rollout_shared(
    game: SoccerGame,
    logits0,
    logits1,
    start_state: State,
    rollout_len: int,
    rng: np.random.Generator,
) -> tuple[list[State], list[int], list[int], list[float], State]:
    """Like :func:`_rollout`, but ``logits0``/``logits1`` may come from a
    (partially) shared network instead of two independent ``PolicyNet``s --
    both are evaluated on the same raw state, no transform of any kind."""
    state = start_state
    states: list[State] = []
    a0s: list[int] = []
    a1s: list[int] = []
    r0s: list[float] = []
    for _t in range(rollout_len):
        x = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            d0 = torch.distributions.Categorical(logits=logits0(x))
            d1 = torch.distributions.Categorical(logits=logits1(x))
        a0 = int(d0.sample())
        a1 = int(d1.sample())
        ns, (r0, _r1), done = game.step(
            state, MOVE_ACTIONS[a0], MOVE_ACTIONS[a1], rng=rng
        )
        states.append(state)
        a0s.append(a0)
        a1s.append(a1)
        r0s.append(r0)
        state = game.initial_state() if done else ns
    return states, a0s, a1s, r0s, state


def train_reinforce_selfplay_shared(
    game: SoccerGame,
    architecture: str = "shared",
    gamma: float = 0.9,
    hidden: int = 64,
    iterations: int = 2000,
    rollout_len: int = 100,
    n_rollouts: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
) -> ReinforceResult:
    """Self-play REINFORCE with a network-sharing architecture, instead of
    :func:`train_reinforce_selfplay`'s two fully independent ``PolicyNet``s
    -- the "network architecture choices" question from the meeting: does
    sharing weights between the two players' policies change self-play
    training?

    Deliberately does **not** use the game's left-right mirror symmetry to
    construct player 1's policy from player 0's, the way an earlier version
    of this function did. A symmetric game is not guaranteed to have only
    symmetric equilibria -- baking the symmetry into the network's
    parameterization presupposes the answer to a question self-play is
    supposed to be free to discover on its own, and would systematically
    rule out any genuinely asymmetric equilibrium the unconstrained game
    might actually have. Both architectures below feed the *same raw joint
    state* to both players, identically -- whatever relationship the two
    learned policies end up with is something training found, not something
    the architecture assumed going in.

    ``architecture="shared"``: one :class:`JointPolicyNet` -- a single
    shared body all the way to an 8-logit output (4 per player), both
    policies read off the same forward pass on the same input.

    ``architecture="partial"``: a :class:`SharedTrunkPolicyNet` -- shared
    body, separate linear head per player, both fed the same raw state.

    Not a drop-in replacement for :func:`train_reinforce_selfplay` (which
    stays exactly as it was, fully independent nets) -- a separate function
    so that existing "separate" results and tests are untouched by this."""
    if architecture not in ("shared", "partial"):
        raise ValueError(f"architecture must be 'shared' or 'partial', got {architecture!r}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if architecture == "shared":
        net: nn.Module = JointPolicyNet(hidden)
        logits0 = net.logits0
        logits1 = net.logits1
    else:
        net = SharedTrunkPolicyNet(hidden)

        def logits0(x: torch.Tensor) -> torch.Tensor:
            return net(x, player=0)

        def logits1(x: torch.Tensor) -> torch.Tensor:
            return net(x, player=1)

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    trace: list[float] = []
    state = game.initial_state()
    for _ in range(iterations):
        states: list[State] = []
        a0s: list[int] = []
        a1s: list[int] = []
        g0s: list[float] = []
        raw_r0: list[float] = []

        for k in range(n_rollouts):
            start = state if k == 0 else game.initial_state()
            s_k, a0_k, a1_k, r0_k, end_state = _rollout_shared(
                game, logits0, logits1, start, rollout_len, rng,
            )
            if k == 0:
                state = end_state
            states += s_k
            a0s += a0_k
            a1s += a1_k
            g0s += _discounted_returns(r0_k, gamma)
            raw_r0 += r0_k

        g1s = [-g for g in g0s]

        X = torch.tensor(np.array(states), dtype=torch.float32)
        A0 = torch.tensor(a0s, dtype=torch.long)
        A1 = torch.tensor(a1s, dtype=torch.long)
        G0 = torch.tensor(g0s, dtype=torch.float32)
        G1 = torch.tensor(g1s, dtype=torch.float32)

        logp0 = torch.log_softmax(logits0(X), dim=-1).gather(1, A0[:, None]).squeeze(1)
        logp1 = torch.log_softmax(logits1(X), dim=-1).gather(1, A1[:, None]).squeeze(1)
        # One combined loss, one backward pass: both players' gradients flow
        # into the same shared parameters (all of them for "shared", the
        # trunk for "partial") in a single step.
        loss = -(logp0 * G0).mean() - (logp1 * G1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        trace.append(float(np.mean(raw_r0)))

    net0_view = _HeadPolicyView(logits0)
    net1_view = _HeadPolicyView(logits1)
    return ReinforceResult(net0=net0_view, net1=net1_view, mean_reward_trace=trace)
