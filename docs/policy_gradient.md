# Self-play REINFORCE: does policy gradient converge to a Nash equilibrium?

> September 28 audit: PG results below predate the terminal-return fix and
> are historical, not validation of the corrected episodic trainer. The new
> two-seed reward comparison is in [reward-q-audit.md](reward-q-audit.md).
> The explorer's representative PG predictions have been regenerated with
> terminal boundaries. DQN results are unaffected by that PG defect.


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
* **shared** -- one `JointPolicyNet`: a single shared body all the way to
  an 8-logit output (4 per player), both policies read off the same
  forward pass on the *same raw joint state*. One optimizer; both players'
  gradients land on the same weights every step.
* **partial** -- `SharedTrunkPolicyNet`: only the trunk (geometry feature
  extraction) is shared; each player keeps its own final linear head, both
  fed the same raw state.

Neither architecture uses the game's own left-right mirror symmetry to
construct one player's policy from the other's. An earlier version of
this experiment did exactly that for `shared` (query the net at the
mirrored state, flip L/R) -- flagged as invalid on exactly the right
grounds: **a symmetric game is not guaranteed to have only symmetric
equilibria**, so building that symmetry into the architecture presupposes
the answer to the question this experiment exists to test, and would rule
out any genuinely asymmetric equilibrium the unconstrained game might
actually settle on. Corrected here: both architectures now read the
identical raw state for both players, with no transform of any kind, and
let training find whatever relationship between the two policies it
finds.

| (5 seeds) | row agreement | col agreement | exploitability |
|---|---|---|---|
| separate | 0.333 ± 0.157 | 0.510 ± 0.070 | 0.863 ± 0.064 |
| shared | 0.291 ± 0.127 | 0.483 ± 0.048 | 0.904 ± 0.060 |
| partial | 0.390 ± 0.277 | 0.482 ± 0.013 | 0.897 ± 0.060 |

The qualitative finding survives the correction: **both sharing
architectures are still more exploitable on average than two fully
independent nets** (0.904 and 0.897 vs. 0.863), so this was not an
artifact of the earlier flawed symmetry trick. What *did* change is
everything about the variance and the row/column split. With no
symmetry constraint, `shared`'s row and column agreement are no longer
forced close to each other (0.291 vs. 0.483 -- genuinely different
policies, confirmed directly by
`test_shared_architecture_does_not_impose_mirror_equivariance`, which
checks the two are *not* equal rather than that they are). `partial`'s
row agreement now has the widest spread of any architecture in this
comparison (± 0.277) -- one seed reached 0.704 while three others sat at
0.188-0.189, a genuinely bimodal outcome that the old, symmetry-biased
version never showed.

The likely mechanism for the headline result is unchanged: self-play is
supposed to model two *independent* optimizers, each implicitly
best-responding to the other's current policy. Tying the two players'
parameters together means every gradient step moves *both* players'
effective policies at once, through the same weights, in a single
combined update -- there is no "player 1's policy held fixed while
player 0 improves" moment, which is closer to what two separate networks
approximate. Forcing a shared representation onto two adversaries doesn't
just save parameters -- it removes some of the independence self-play's
convergence intuition relies on, and here that still costs more than the
parameter sharing saves, whether or not the architecture also happens to
assume the equilibrium is symmetric.

## A learned baseline and an entropy bonus, tested separately

Two more levers, deliberately not bundled together: a **learned value
baseline** (`advantage = G - V(s)`, `V` a small regression net trained
toward the observed return) is pure variance reduction -- it does not
change what the expected gradient points toward, only how noisy each
sample estimate is. An **entropy bonus** (`-entropy_coef * H(pi(.|s))`
added to the loss) is a different mechanism entirely: it directly rewards
a less-peaked policy, targeting premature collapse to near-determinism,
not variance as such. Four arms per seed -- neither, baseline only,
entropy only, both -- so an effect can be attributed to one mechanism, the
other, or their combination
(`train_reinforce_selfplay(..., use_baseline=, entropy_coef=)`,
`scripts/policy_gradient_ablation.py`).

Same setup as the from-scratch baseline (2000 iterations x 100 steps),
**3 seeds**, `entropy_coef=0.1`, `experiments/policy_gradient_ablation.csv`:

| (3 seeds) | row agreement | col agreement | exploitability |
|---|---|---|---|
| neither | 0.308 ± 0.113 | 0.494 ± 0.088 | 0.841 ± 0.079 |
| baseline only | 0.240 ± 0.085 | 0.495 ± 0.055 | 0.845 ± 0.045 |
| entropy only | 0.347 ± 0.085 | 0.391 ± 0.036 | 0.833 ± 0.007 |
| both | 0.293 ± 0.070 | 0.401 ± 0.028 | 0.819 ± 0.009 |

Neither lever fixes convergence -- all four arms land in the same
0.24-0.35 row-agreement range, nowhere close to what pre-training reaches.
But they are not doing the same thing. **The baseline alone barely moves
anything**: mean exploitability does not improve (0.841 -> 0.845), and its
variance reduction is modest (±0.079 -> ±0.045) -- not the dramatic effect
"pure variance reduction" might suggest, though 3 seeds is too few to rule
out noise here. **The entropy bonus alone is where the real effect
shows up, and it is a consistency effect, not a convergence effect**:
exploitability's standard deviation drops more than tenfold (±0.079 ->
±0.007) and its mean improves slightly (0.841 -> 0.833), while row
agreement does not clearly improve (0.308 -> 0.347, within noise) and col
agreement drops (0.494 -> 0.391). Entropy's job here is making the
*outcome* far more reproducible seed-to-seed, not making self-play find
the equilibrium more often. Combining both gives the best mean
exploitability (0.819) and keeps entropy's low variance (±0.009), but the
gain over "neither" is incremental, not transformative -- consistent with
this page's running theme: every lever tried so far changes *how
consistently* self-play lands somewhere, far more than it changes *where*.

## Does batching help with a fair update budget?

The batching section above flagged the natural follow-up: the original
comparison matched *total environment steps* (200,000 either way), which
gave the batched arm 8x fewer gradient *updates* (250 vs. 2000) -- exactly
the condition the "too few updates to differentiate a state-independent
direction into a state-conditional policy" mechanism needs. Matching
*update count* instead (`scripts/policy_gradient_batch.py --match
updates`) keeps both arms at 2000 iterations, so the batched arm now sees
8x the single arm's total data (1,600,000 vs. 200,000 steps) -- an
unequal-data comparison, on purpose, to isolate update count as the one
variable in question.

**5 seeds**, `experiments/policy_gradient_batch_updates.csv`:

| (5 seeds) | row action agreement | exploitability |
|---|---|---|
| 1 rollout x 2000 iterations (200,000 steps) | 0.333 ± 0.157 | 0.863 ± 0.064 |
| 8 rollouts x 2000 iterations (1,600,000 steps) | 0.300 ± 0.126 | 0.831 ± 0.041 |

Update-count-matched batching does **not** collapse to a state-independent
policy -- row agreement varies genuinely across seeds (0.165 to 0.458, not
a repeated constant), confirming the mechanism diagnosed earlier: the
collapse was about too few gradient updates, not something inherent to
decorrelated batching. With a fair update budget, batching lands at
roughly the same mean row agreement as the correlated single trajectory
(0.300 vs. 0.333, within the overlap of their error bars) but with
somewhat lower exploitability (0.831 vs. 0.863) and somewhat lower
variance on both metrics. A real but modest improvement, not the dramatic
win the "more independent data should obviously help" intuition might
predict -- and it costs 8x the environment interactions to get it.

## REINFORCE vs. A2C vs. PPO, and fictitious play

The research-group note asks for REINFORCE, A2C and PPO "to replicate the exact
solver", and fictitious play as a side quest. `soccer_nash/actor_critic.py`
adds A2C (n-step bootstrapped advantage, one step per batch) and PPO (GAE
`lambda = 0.95`, clip 0.2, 4 epochs) with the same two-network self-play setup,
the same `+/-1` reward, `gamma = 0.9`, and the same budget (2000 x 100 steps)
as REINFORCE (here with the learned baseline and an entropy bonus of 0.01).
All three are scored by the same evaluator against the exact A10 solve; 3 seeds,
mean over seeds:

| algorithm | exploitability (exact = 0) | mean equilibrium regret | row action agreement | row vs. random | row vs. best response |
|---|---|---|---|---|---|
| REINFORCE (+baseline, +entropy) | 0.834 | 0.124 | 0.281 | +0.161 | -0.393 |
| A2C | 0.830 | 0.122 | 0.292 | +0.127 | -0.403 |
| PPO | 0.874 | 0.167 | 0.253 | +0.170 | -0.371 |

**None of the three gets near the equilibrium**: exploitability stays around
0.83-0.87 (a deterministic offense has a perfect defense, so the worst case is
close to -1), and the differences between algorithms are within seed noise
(3 seeds; PPO's column agreement ranges 0.14-0.52). That is the same result as
the REINFORCE sections above, so a better advantage estimator or a clipped update
does not fix it. The A2C/PPO hyperparameters were **not tuned** (see
[assumptions.md](assumptions.md), S8): this shows these defaults fail, not that
the algorithms cannot succeed.

**Fictitious play** (`soccer_nash/fictitious_play.py`) behaves as the note
sketches. On rock-paper-scissors, best-response dynamics cycle through the three
diagonal joint actions forever, while fictitious play converges to
`(1/3, 1/3, 1/3)` (value bracket +/-0.004 at 10,000 rounds). Run on all 2,380
stage games of the random move-order board (fixed exact `V*`, 94 of them with no
pure saddle), the value error is below 0.0014 after 1,000 rounds and below
0.0004 after 10,000; the mixed states are slower (mean bracket 0.0018 vs. 0.0002
at 1,000 rounds). This is fictitious play on stage games with the exact
continuation value, not yet a learning dynamic inside the Markov game.

### The same comparison on the random move-order board

The 7x5 random move-order game (three goal rows, 94 mixed stage games) is the
board where the equilibrium actually mixes, so it is the more demanding test.
Same budget, 3 seeds, scored against the exact solve. Win / tie / loss are
Monte-Carlo estimates from the kickoff over 1,000 games each (a tie is a game
with no goal in 100 steps), for the row player only; the "best response" is
the exact best response in *discounted value*, not in win probability.

| row player | exploitability | mean eq. regret | W / T / L vs. random | vs. exact Nash | vs. best response | mirror gap (mean / max) |
|---|---|---|---|---|---|---|
| exact solver (reference) | 0 | 0 | 1.00 / 0.00 / 0.00 | 0.66 / 0.01 / 0.34 | 0.70 / 0.00 / 0.30 | 0 |
| REINFORCE (+baseline, +entropy) | 0.733 | 0.056 | 0.92 / 0.04 / 0.04 | 0.25 / 0.00 / 0.75 | 0.07 / 0.00 / 0.93 | 0.42 / 0.97 |
| A2C | 0.781 | 0.058 | 0.88 / 0.07 / 0.05 | 0.12 / 0.00 / 0.88 | 0.04 / 0.00 / 0.96 | 0.44 / 0.96 |
| PPO | 0.852 | 0.108 | 0.44 / 0.49 / 0.07 | 0.15 / 0.09 / 0.76 | 0.00 / 0.00 / 1.00 | 0.78 / 1.00 |

All three beat a random opponent (PPO mostly ties it) and all three lose
heavily to the exact equilibrium and to a best response, so none has learned the
equilibrium; REINFORCE and A2C are closer than PPO here, though three seeds
cannot rank them. The exact row player wins 66% against the exact column player
only because it starts with the ball. The *mirror gap* is how far the two
separately trained policies are from being mirror images of each other (0 for the
exact solve): 0.4-0.8 on average, so the two networks learn clearly different
strategies, not a symmetric pair.

### Fictitious play inside the Markov game

`markov_fictitious_play` drops the exact continuation value. Every sweep rebuilds
each state's stage matrix from the *current* value estimate and updates it from
fictitious play on that matrix, on the random move-order board against the exact solve:

| variant | after | max value error | mean value bracket |
|---|---|---|---|
| restart beliefs each sweep, 200 rounds | 50 sweeps | 0.0124 | 0.0012 |
| persistent beliefs, one round per sweep | 100 sweeps | 0.083 | 0.0085 |
| persistent beliefs, one round per sweep | 500 sweeps | 0.0165 | 0.0022 |
| persistent beliefs, one round per sweep | 2,000 sweeps | 0.0051 | 0.0006 |

The persistent variant is the learning dynamic of the note (each player
best-responds to the opponent's whole past play, per state). It converges, slowly
and at roughly a `1/sqrt(t)` rate; the restart variant plateaus at the fixed
inexactness of 200 rounds of fictitious play. Unlike the policy-gradient methods,
fictitious play recovers the equilibrium values here, because it solves each
stage game by best-responding to a matrix rather than by sampling.

### Exploring starts, and Nash-DQN for comparison

The policy-gradient runs above train from the kickoff, so most of the 2,380 states are rarely visited (S5 in [solver_assumptions.md](solver_assumptions.md)). `--explore` restarts every episode at a uniformly random state instead. Same board, budget and 3 seeds:

| row player | exploitability | mean eq. regret | row action agreement | W / T / L vs. random | vs. exact Nash | vs. best response | mirror gap (mean / max) |
|---|---|---|---|---|---|---|---|
| REINFORCE, exploring starts | 0.801 | 0.027 | 0.710 | 0.92 / 0.07 / 0.01 | 0.06 / 0.00 / 0.95 | 0.00 / 0.00 / 1.00 | 0.11 / 0.79 |
| A2C, exploring starts | 0.821 | 0.030 | 0.698 | 0.89 / 0.09 / 0.02 | 0.05 / 0.00 / 0.95 | 0.00 / 0.00 / 1.00 | 0.15 / 0.81 |
| PPO, exploring starts | 0.910 | 0.072 | 0.509 | 0.90 / 0.05 / 0.06 | 0.02 / 0.00 / 0.98 | 0.01 / 0.00 / 0.99 | 0.41 / 0.99 |
| Nash-DQN (minimax of its own `Q`) | 0.378 | -- | -- | 0.995 / 0.004 / 0.001 | 0.52 / 0.10 / 0.39 | 0.02 / 0.79 / 0.19 | 0.39 / 1.00 |
| exact solver | 0 | 0 | 1 | 1.00 / 0.00 / 0.00 | 0.66 / 0.01 / 0.34 | 0.70 / 0.00 / 0.30 | 0 |

Exploring starts do what they should on average: action agreement rises from 0.44 to 0.71 for REINFORCE, mean equilibrium regret halves, and the two networks become much closer to mirror images (0.42 to 0.11). They do **not** fix play from the kickoff: exploitability and the results against the exact equilibrium get slightly worse, so being right at more states is not the same as being unexploitable at the one that matters. Nash-DQN, which fits the exact `Q` matrix instead of sampling returns, is far closer to the exact solver on every measure. That is the same ordering as [neural.md](neural.md).

### Best-response dynamics, and what the note's "???" resolves to

With the standard payoffs, the best response to a half-and-half mix of rock and scissors is Rock (expected `+1/2`; Paper scores 0, Scissors `-1/2`), so the note's `br(1/2 R, 1/2 S)` is Rock. Of the 87 random-board states where the equilibrium actually mixes, 79 mix two actions and 8 mix three, the case the note sketches as `1/3, 1/3, 1/3`.

### Fictitious play as a policy, not only a value

Beyond the value error above, the final empirical mixes were scored as policies (2000 sweeps persistent, or 150 sweeps of 200 rounds restarted):

| board | variant | mean eq. regret | max eq. regret | exploitability | row mix distance to exact, 94 mixed states |
|---|---|---|---|---|---|
| random | persistent | 0.0006 | 0.008 | 0.011 | 0.128 |
| random | restart | 0.0011 | 0.009 | 0.025 | 0.159 |
| deterministic | persistent | 0.0005 | 0.010 | 0.006 | -- (no mixed states) |
| deterministic | restart | 0.0015 | 0.016 | 0.027 | -- |

Fictitious play gets close to an equilibrium as a whole (exploitability about 0.01, against 0.4-0.9 for every learner above), but its empirical mixes at the mixed states are still about 0.13 off in total variation from the exact mix. Some of those states have several equilibria, so that distance is partly a comparison against one arbitrary choice.

## Reproducing

```
python scripts/policy_gradient.py --seeds 5              # from-scratch only
python scripts/policy_gradient_warmstart.py --seeds 3     # the three-way comparison
python scripts/policy_gradient_batch.py --seeds 5         # single trajectory vs. batched rollouts (steps-matched)
python scripts/policy_gradient_batch.py --seeds 5 --match updates  # same, update-count-matched
python scripts/policy_gradient_architectures.py --seeds 5 # separate vs. shared vs. partial-share nets
python scripts/policy_gradient_ablation.py --seeds 3       # learned baseline vs. entropy bonus, separately
python scripts/pg_algos_and_fp.py pg --seeds 3             # REINFORCE vs. A2C vs. PPO
python scripts/pg_algos_and_fp.py pg --board random --seeds 3   # same, on the random move-order board
python scripts/pg_algos_and_fp.py pg --board random --explore --seeds 3   # exploring starts
python scripts/pg_algos_and_fp.py dqn --seeds 3             # Nash-DQN win rates
python scripts/pg_algos_and_fp.py fp                       # best-response dynamics vs. fictitious play
python scripts/pg_algos_and_fp.py markov-fp                # fictitious play inside the Markov game
```

`soccer_nash/policy_gradient.py`'s `pretrain_policy_nets` and
`train_reinforce_selfplay(..., init_net0=, init_net1=)` are the two building
blocks, plus `train_reinforce_selfplay(..., n_rollouts=)` for the batched
comparison and `train_reinforce_selfplay_shared(..., architecture=)` (with
`JointPolicyNet`/`SharedTrunkPolicyNet`) for the weight-sharing comparison;
`tests/test_policy_gradient.py` checks the `init_net` seeding property
directly (mirroring `nash_dqn.py`'s own `init_net` test), that a
best-responder can never do better than a random opponent for any policy,
that a rollout boundary is never treated as a continuation of another
rollout's trajectory when computing returns, and that the `shared`
architecture does *not* impose mirror-equivariance between the two
players' policies.
