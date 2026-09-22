# Self-play REINFORCE: does policy gradient converge to a Nash equilibrium?

The Sept 17 meeting's own question, verbatim -- policy gradient "is not
directly solving \[the] game anywhere... can this converge?" -- plus its own
follow-up, left open at the time ("would pre-training help? ... minor
issue"). Both answered empirically here, the same way [neural.md](neural.md)
answers the analogous question for Q-learning: never compared to the other
player's net, always to the exact solver
(`soccer_nash/policy_gradient.py`, `NashQIteration.run_exact()`).

Two independent `PolicyNet`s, one per player, trained by plain REINFORCE (no
baseline, no entropy bonus -- the meeting's own derivation has neither, so
this implementation adds neither on its own; see the module docstring) on
genuine on-policy self-play: real `game.step()` samples, not the exact
transition distribution `nash_dqn.py` uses, because this is the actual
policy-gradient algorithm, not a fitted-Q stand-in.

## From a random init, self-play alone does not get close

On the deterministic A10 board, over **5 seeds**, 2000 iterations of
100-step self-play rollouts each (`experiments/policy_gradient_seeds.csv`,
`make policy-gradient`):

| (5 seeds) | row action agreement | exploitability (duality gap) | vs. random | vs. best response |
|---|---|---|---|---|
| self-play REINFORCE | 33% ± 16% | 0.86 ± 0.06 | +0.24 ± 0.16 | -0.41 ± 0.04 |

Wide seed-to-seed variance (agreement ranges 19%-56% across the 5 seeds) and
an exploitability close to the worst possible for this board. It beats a
random opponent on average (`+0.24`) -- it did learn *something* -- but a
best responder that has seen its actual mixing probabilities turns that into
a clear loss (`-0.41`), consistently across all 5 seeds. **Plain self-play
REINFORCE, from scratch, does not reliably find anything close to the exact
equilibrium** in this many iterations -- the meeting's own worry, confirmed
rather than assumed.

## Does pre-training help? Yes to start, then self-play freezes it exactly where it started

Three starting points, the same shape as `nash_dqn.py`'s from-zero /
fit-to-exact / warm-start comparison (`scripts/policy_gradient_warmstart.py`,
**3 seeds**, `experiments/policy_gradient_warmstart.csv`):

* **from scratch** -- random init, straight into 2000 iterations of self-play.
* **fit to exact** -- supervised cross-entropy regression onto the exact
  equilibrium policies (`pretrain_policy_nets`), no rollouts, no self-play.
* **warm start** -- the fit-to-exact weights, then the *same* 2000 iterations
  of self-play as "from scratch".

| (3 seeds) | row action agreement | exploitability (duality gap) |
|---|---|---|
| from scratch | 0.308 ± 0.113 | 0.841 ± 0.079 |
| fit to exact | 0.976 ± 0.007 | 0.828 ± 0.011 |
| warm start | 0.976 ± 0.007 | 0.828 ± 0.011 |

**Fit-to-exact and warm-start are identical to three decimal places, in
every one of the 3 seeds, on both metrics.** Not "the warm start mostly
holds" the way `nash_dqn.py`'s Q-net warm-start does (§ [neural.md](neural.md),
where 2000 more epochs of TD bootstrap visibly moves the metrics) -- here
2000 *more* iterations of live self-play leave the policy numerically
unchanged. The mechanism is not mysterious: `pretrain_policy_nets` fits the
net close enough to the exact equilibrium that its softmax output is already
sharply peaked at most states, and REINFORCE's gradient is a *score-function*
estimate, `∇log π(a|s) · G` -- for a near-deterministic `π`, the sampled
action is almost always the same one, so the gradient signal (and its
variance) collapses toward zero. With no entropy bonus to keep the policy
exploring and no baseline to reduce variance on top of that, there is
essentially nothing left to push the weights anywhere once pre-training gets
close.

> **Self-play REINFORCE does not converge here on its own, but pre-training
> plus self-play is not a fix either -- it is a fit-to-exact result with
> extra steps.** The 2000 iterations of further training neither help nor
> hurt; they are computationally inert once the policy is peaked. If this
> project revisits policy gradient, the natural next step -- per the
> meeting's own note that a baseline "is the natural next step if variance
> turns out to be the bottleneck" -- is exactly that: an entropy bonus or a
> learned baseline to keep the gradient alive, not more iterations of the
> same loop.

## Does batching independent rollouts help? It cuts variance -- by collapsing to the same degenerate policy every time

The 100-step rollouts above already reuse every suffix of their own single
trajectory as a training pair (`_discounted_returns`), but those T pairs
come from *one* trajectory, so they are correlated, not i.i.d. -- each
state depends on the last. The natural further step: collect several
*independent* rollouts per iteration (each restarted at `game.initial_state()`)
and batch them together into one gradient step, computing each rollout's
discounted returns separately so a boundary between two rollouts is never
mistaken for a continuation of one (`train_reinforce_selfplay`'s new
`n_rollouts` argument, `scripts/policy_gradient_batch.py`).

Same total environment-step budget for both arms (200,000 steps), **5
seeds**, `experiments/policy_gradient_batch.csv`:

| (5 seeds) | row action agreement | exploitability (duality gap) |
|---|---|---|
| 1 rollout x 2000 iterations (correlated) | 0.333 ± 0.157 | 0.863 ± 0.064 |
| 8 rollouts x 250 iterations (independent, batched) | 0.188 ± 0.000 | 0.860 ± 0.007 |

Variance across seeds does collapse, dramatically -- row agreement's
standard deviation goes from 0.157 to 0.000 (**not rounding**: all five
seeds print `0.1878` to four decimal places). But the mean does not
improve, and inspecting the trained nets directly shows why: every one of
the 5 batched-rollout networks converges to predicting the **same single
action, regardless of the state it is looking at** -- a completely
state-independent policy -- while the correlated single-trajectory nets
stay state-dependent (mixing between two or more actions across states,
seed to seed). The 0.1878 is not five runs converging near the equilibrium
together; it is five runs converging to the *same trivial constant policy*,
whose fixed agreement with the exact solver's mix of pure actions happens
to be 18.78% regardless of which seed produced it.

The mechanism: 250 large-batch (800-sample) gradient steps is an order of
magnitude fewer *updates* than 2000 small ones, even though both see the
same total data. A large, low-variance batch gradient reliably points
REINFORCE toward whichever single action has the best average return
*across the whole state distribution it was sampled from* -- exactly the
state-independent direction -- and with so few updates, the net never gets
enough gradient steps to differentiate that into a state-conditional
policy. The noisier single-trajectory version, despite (or because of) its
higher variance, keeps enough per-update signal tied to *specific* states
to avoid collapsing all the way to a constant. **Decorrelating the batch
reduces variance, but at this fixed compute budget it reduces variance by
finding a worse, simpler optimum faster, not by converging more reliably
toward the real one.** Matching update *count* (2000 iterations at 8
rollouts each, 16x the environment steps) rather than update *count x
env-steps-per-update*, or shrinking the per-update batch, would be the
natural next thing to try if this is revisited.

## Does it matter how much the two players' networks share weights?

Everything above uses two fully independent `PolicyNet`s -- the meeting's
other open design question was whether that is even the right choice for a
two-player self-play setup, versus sharing weights between the players.
Three architectures, same training budget (2000 iterations x 100 steps),
same 5 seeds, `train_reinforce_selfplay_shared`,
`scripts/policy_gradient_architectures.py`:

* **separate** -- the baseline above: two independent nets.
* **shared** -- *one* `PolicyNet`. Player 1's policy at state `s` is not a
  second output of that network; it is *defined* as the network's own
  mirror-image policy -- query it at the board's left-right mirror image
  and swap L/R in the result (`soccer_nash/symmetry.py`'s proven
  equilibrium anti-symmetry, used here as a parameterization, not assumed
  to hold mid-training). One optimizer; both players' gradients land on the
  same weights every step.
* **partial** -- `SharedTrunkPolicyNet`: the same mirrored-input scheme,
  but only the trunk (geometry feature extraction) is shared; each player
  keeps its own final linear head.

| (5 seeds) | row agreement | col agreement | exploitability |
|---|---|---|---|
| separate | 0.333 ± 0.157 | 0.510 ± 0.070 | 0.863 ± 0.064 |
| shared | 0.373 ± 0.243 | 0.373 ± 0.242 | 0.949 ± 0.023 |
| partial | 0.364 ± 0.136 | 0.467 ± 0.104 | 0.907 ± 0.011 |

Sharing does not help here, on either axis it was meant to help. Weight
sharing was expected to be a variance reducer (more effective data per
parameter, one net sees both players' rollouts) -- instead **`shared` has
the *highest* row-agreement variance of the three** (± 0.243, worse than
two fully independent nets), and both `shared` and `partial` are *more*
exploitable on average than `separate` (0.949 and 0.907 vs. 0.863). The one
thing sharing reliably buys is internal consistency, not quality: `shared`'s
row and column agreement are within noise of each other in every seed
(0.1878/0.1887, 0.6538/0.6538, ...) because they are, by construction, the
same function evaluated at mirrored inputs -- confirmed directly
(`test_shared_architecture_satisfies_mirror_equivariance_by_construction`),
not just approximately similar.

The likely mechanism: self-play is supposed to model two *independent*
optimizers, each implicitly best-responding to the other's current policy.
Tying the two players' parameters together means every gradient step moves
*both* players' effective policies at once, through the same weights, in a
single combined update -- there is no "player 1's policy held fixed while
player 0 improves" moment, which is closer to what two separate networks
(each updated only by its own loss, but from the *other* player's current,
just-sampled behaviour) approximate. Forcing a shared representation onto
two adversaries doesn't just save parameters -- it removes some of the
independence self-play's convergence intuition relies on, and here that
costs more than the parameter sharing saves. `partial`, which keeps the two
heads independent, lands between the two extremes on every metric,
consistent with that story.

## Reproducing

```
python scripts/policy_gradient.py --seeds 5              # from-scratch only
python scripts/policy_gradient_warmstart.py --seeds 3     # the three-way comparison
python scripts/policy_gradient_batch.py --seeds 5         # single trajectory vs. batched rollouts
python scripts/policy_gradient_architectures.py --seeds 5 # separate vs. shared vs. partial-share nets
```

`soccer_nash/policy_gradient.py`'s `pretrain_policy_nets` and
`train_reinforce_selfplay(..., init_net0=, init_net1=)` are the two building
blocks, plus `train_reinforce_selfplay(..., n_rollouts=)` for the batched
comparison and `train_reinforce_selfplay_shared(..., architecture=)` (with
`SharedTrunkPolicyNet`) for the weight-sharing comparison;
`tests/test_policy_gradient.py` checks the `init_net` seeding property
directly (mirroring `nash_dqn.py`'s own `init_net` test), that a
best-responder can never do better than a random opponent for any policy,
that a rollout boundary is never treated as a continuation of another
rollout's trajectory when computing returns, and that the `shared`
architecture's mirror-equivariance holds exactly, not approximately.
