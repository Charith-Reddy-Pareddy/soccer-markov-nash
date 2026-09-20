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

## Reproducing

```
python scripts/policy_gradient.py --seeds 5              # from-scratch only
python scripts/policy_gradient_warmstart.py --seeds 3     # the three-way comparison
```

`soccer_nash/policy_gradient.py`'s `pretrain_policy_nets` and
`train_reinforce_selfplay(..., init_net0=, init_net1=)` are the two building
blocks; `tests/test_policy_gradient.py` checks the `init_net` seeding
property directly (mirroring `nash_dqn.py`'s own `init_net` test) and that a
best-responder can never do better than a random opponent, for any policy.
