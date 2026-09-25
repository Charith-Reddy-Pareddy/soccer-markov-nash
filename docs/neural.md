# Neural Nash-Q: full ablation detail

Groundwork for the eventual continuous-action work, not an equal-weight
contribution to [report.md](report.md) -- the exact solver stays the ground
truth throughout. This page has the full tables and prose behind report.md
§8's summary; start there for the two headline findings ("action accuracy
is not equilibrium accuracy" and the capacity/exploitability split) and come
here for every number behind them.

## Three starting points for the Q net, and where the approximation breaks

On the deterministic 7×5 game, over **5 seeds** (`experiments/nash_dqn_seeds.csv`):
a **Q net** regresses toward the stage matrices `Q(s, a0, a1)` and extracts a
policy by minimax; a **policy net** regresses *directly* onto the exact
equilibrium strategies `(p(s), q(s))`. The Q net is tried from three starting
points, all the same architecture and all scored the same way: **from zero**
(random init, pure TD bootstrap, 600 epochs), **fit to exact** (supervised
regression straight onto `Q_exact`, no bootstrap at all, 600 epochs), and
**warm start** (the fit-to-exact weights, then the same 600-epoch TD
bootstrap as "from zero").

| (5 seeds) | exact | from zero | fit to exact | warm start | policy net |
|---|---|---|---|---|---|
| max `\|V − V_exact\|` | 0 | 0.48 ± 0.03 | 0.38 ± 0.05 | 0.43 ± 0.05 | — |
| action agreement | 100% | 42% ± 4% | 45% ± 2% | 48% ± 4% | **99.9% ± 0.0%** |
| pure/mixed classification | 100% | 68% ± 2% | 47% ± 2% | 65% ± 2% | — |
| max entrywise error `\|Q_θ − Q*\|` | 0 | 0.88 ± 0.07 | 1.21 ± 0.05 | 0.93 ± 0.05 | — |
| max equilibrium regret `ε_NE` | 0 | 0.83 ± 0.07 | 0.77 ± 0.17 | 0.55 ± 0.11 | 0.46 ± 0.02 |
| exploitability (duality gap) | `<1e-9` | 0.44 ± 0.07 | 0.47 ± 0.04 | **0.36 ± 0.07** | 0.84 ± 0.00 |

**Fitting the Q net directly to `Q_exact` -- no bootstrap, no target network,
nothing but supervised regression against the ground truth -- still only
reaches 45% action agreement, barely above the 42% a random init gets from
600 epochs of TD bootstrap.** That is the headline result of the warm-start
ablation: the bottleneck is not bootstrap noise. A network of this size,
extracting a policy via minimax of its own predicted matrix, cannot represent
`Q_exact` closely enough for the *argmax* to survive, even when it is simply
told the answer -- confirmed directly, not just inferred from the value error,
by the raw entrywise gap: fitting straight onto `Q_exact` with nothing but
supervised regression still leaves a worst-case cell off by `1.21`, *more*
than either bootstrap-trained variant, because minimizing squared error
across all 16 cells at once is a different objective than getting the
`argmax` right at any single one. Continuing training from that fit-to-exact
starting point (warm start) buys back real ground on every other axis:
action agreement rises to 48% (from 45%), pure/mixed classification recovers
from 47% back to 65% (against from-zero's 68%), equilibrium regret drops
from 0.77 to 0.55, and exploitability drops to 0.36 -- the best of the
three, beating both from-zero (0.44) and fit-to-exact alone (0.47) in 4 of 5
seeds. So the warm start does not fix the underlying representational
problem (its own entrywise error, 0.93, sits between the other two, not
below both), but it is not wasted either: the extra bootstrapped training
pulls the network back toward respecting the game's saddle structure without
giving up all of the head start on value error.

**The policy net names the exact-optimal action at 99.9% of states and is
still *more* exploitable than any of the three Q nets** (duality gap 0.84 vs
0.36-0.47). Two reasons: the ~0.1% wrong states are exactly the ones a
best-responder attacks (the rock-paper-scissors trap again), and a softmax
head necessarily smears what should be pure strategies into exploitable
near-indifference. So the failure is not primarily value approximation --
"name the right action" is not "play a Nash", and neither is "be told
`Q_exact` outright". Even on a small discrete game where the exact solution
is a 0.3 s computation, a straightforward neural approximation does not
preserve game-theoretic robustness. Keep the exact solver as ground truth.

> **Action accuracy is not equilibrium accuracy.** A network that names the
> textbook-optimal move nearly every time can still be trivially exploited,
> because the states where it is wrong are not random -- they are exactly
> the ones a best-responder is looking for. Simple single-agent RL that
> optimizes for "predict the right action" produces a policy that looks
> excellent on that metric and is not robust in the game-theoretic sense at
> all; this is the discrete, measured version of that concern.

## The harder game: random move order, genuinely mixed equilibria

Everything above runs on `A10SoccerGame` -- the **deterministic** game,
where every stage game already has a pure saddle. That made the
from-zero/fit-to-exact/warm-start comparison a genuine function-approximation
study, but not yet an answer to the harder question:

> Can neural Nash-Q reproduce the exact solver on the game where mixed
> equilibria actually matter?

Two real fixes were needed in `soccer_nash/nash_dqn.py`, not just a
different game object, since the deterministic-only version was silently
wrong on this one: `train_nash_dqn`'s bootstrap target used to be the
vectorised *maximin* of the target network's predicted matrix -- exact only
when every stage game has a pure saddle -- and the transition precompute
used `game.step`, which *samples* one outcome, exact only when a joint
action has a single outcome. Both are now general: `_minimax_batch` is the
same pure-fast-path check, vectorised, falling back to the LP only for the
predicted-mixed matrices; `game.transitions` supplies the full outcome
distribution, averaged exactly. `scripts/nash_dqn_random.py` reruns all four
baselines on the same plain 4x4 board as `tournament4.py`/
`tournament_deepdive.py` (5x4, two-cell goals, **random move order**), where
56 of 760 states (7.4%) have no pure saddle -- a game the deterministic-game
experiment never actually exercised (5 seeds, 600 epochs;
`experiments/nash_dqn_random_seeds.csv`):

| (5 seeds) | exact | from zero | fit to exact | warm start | policy net |
|---|---|---|---|---|---|
| action agreement | 100% | 57% ± 3% | 60% ± 2% | 59% ± 2% | **95.8% ± 0.4%** |
| exploitability | `<1e-9` | 0.36 ± 0.04 | **1.01 ± 0.27** | 0.36 ± 0.08 | 1.20 ± 0.01 |
| mean Frobenius error | 0 | 0.42 ± 0.02 | **0.32 ± 0.01** | 0.39 ± 0.02 | — |
| mean equilibrium regret | 0 | 0.014 ± 0.001 | 0.015 ± 0.002 | **0.011 ± 0.001** | — |

**On this board, fitting directly to `Q_exact` is the best *fit* and the
worst *policy*.** Fit-to-exact has the lowest matrix error (0.32) and the
highest action agreement of the three Q nets (60%) -- it is, numerically,
the closest to the exact answer -- and yet its exploitability (1.01) is
roughly 3x *worse* than from-zero's (0.36). Minimizing per-state MSE with no
regard for the game's structure can produce a matrix that is uniformly close
to correct and still names an exploitable policy, because the states that
matter for exploitability are not weighted any differently than the states
that don't. TD bootstrapping, even though it fits the matrix values less
precisely, produces a policy that is harder to punish -- the bootstrap target
itself depends on downstream states' *minimax* values, which couples
consistency across states in a way independent per-state regression cannot.
**Warm-starting recovers almost all of it**: continuing training from the
fit-to-exact weights lands at from-zero's exploitability (0.36) while
keeping better action agreement (59% vs 57%) and the best regret of the
three (0.011) -- on this harder game, unlike the deterministic one, warm
start is the best of the three Q-net variants outright, not just a partial
recovery.

**The policy net's own failure is starker here than on the deterministic
game.** 95.8% action agreement is far better than any Q net's, and its
exploitability (1.20) is still the worst of the four -- the same "action
accuracy is not equilibrium accuracy" lesson, now demonstrated on a game
that genuinely requires randomization rather than one where the right
answer always happens to be a single action.

## Width/depth ablation: is 45% a capacity limit?

Back on the deterministic game (`A10SoccerGame`, where the warm-start result
above was also measured): more *training* doesn't move action agreement
there -- fitting straight to `Q_exact` with zero bootstrap noise lands at
the same ~45% as 600 epochs of TD bootstrap from scratch. The remaining
question is whether more *capacity* does.
`scripts/nash_dqn_ablation.py` sweeps width (2 hidden layers, 32-512 wide --
extended to 512 specifically so the network is not artificially undersized)
and depth (1-4 hidden layers, width 96 fixed) using the fit-to-exact method
above (2 seeds/config, 600 epochs; `experiments/nash_dqn_ablation.csv`):

| width sweep (depth 2) | 32 | 64 | 96 | 128 | 192 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| params | 1.8k | 5.6k | 11.4k | 19.3k | 41.3k | 71.4k | 273.9k |
| action agreement | 43% | 47% | 45% | 45% | 46% | 45% | 47% |
| exploitability | 0.79 | 0.56 | 0.47 | 0.41 | 0.38 | 0.25 | 0.18 |

| depth sweep (width 96) | 1 | 2 | 3 | 4 | 3, width 256 | 3, width 512 |
|---|---|---|---|---|---|---|
| params | 2.1k | 11.4k | 20.8k | 30.1k | 137.2k | 536.6k |
| action agreement | 44% | 45% | 46% | 48% | **48.1%** | 47.5% |
| exploitability | 0.82 | 0.47 | 0.37 | 0.25 | 0.08 | **0.05** |

**Action agreement plateaus -- and stays plateaued all the way to 115x the
original parameter count.** The best action agreement in the whole grid is
depth-3/width-256 (137k params, 48.1%); depth-3/width-512 (537k params, 3.9x
more parameters than the best config, 115x the original `5->96->96->16`
architecture) does not beat it -- it is slightly *lower*, 47.5%. Nowhere in
this grid, including its largest member, does capacity buy a path toward the
policy net's 99.9%: this is not an expressiveness problem in the sense of
"the network can't fit the numbers, give it more room" -- more room stops
mattering for this metric long before the grid runs out.
**Exploitability and value error, on the other hand, never stop improving
with capacity** -- exploitability falls monotonically from 0.79 (1.8k
params) to 0.05 (537k params, a 15x reduction) with no sign of flattening
out even at the largest size tested, and max value error falls from 0.53 to
0.03 over the same range. A bigger net genuinely keeps getting *closer* to
`Q_exact` in the numbers that matter for robustness; it just stops getting
closer in a way the discrete argmax reflects, well before it stops getting
closer in every other sense. That is the same lesson as the policy net's own
failure, from the other direction: matching a continuous target well (small
MSE) and matching a discrete decision well (correct argmax) are different
objectives, and this architecture is much better at the former than the
latter, regardless of size. **Depth beats width at matched budget**:
depth-3/width-96 (20.8k params) beats width-128/depth-2 (19.3k params, the
nearest matched budget) on every metric -- action agreement, exploitability,
and value error alike -- and depth-4/width-96 (30.1k params) beats
width-256/depth-2 (71.4k params, 2.4x the parameters) outright on all three.
The single best configuration in the grid for action agreement -- depth 3,
width 256, 137k params -- is still 52 points short of the policy net; the
single best for exploitability and value error -- depth 3, width 512, 537k
params -- does not even improve on the 137k-param configuration's action
agreement while using 3.9x the parameters to chase the other two metrics
further.

## Roadmap: this is not the eventual challenge

Everything in this page is discrete: a 4x4 (or 5x5) action grid, a
network that outputs a matrix over it. The actual eventual challenge is a
**continuous-action** version of this game -- a continuous dog-and-sheep /
continuous soccer pursuit problem -- and the point of that future work is
*not* to discretize a continuous action space back down to a grid and reuse
everything above unchanged. The intended path:

```
exact discrete game  -->  neural reproduction of it (this page)  -->
continuous-action stage-game solver
```

The discrete work above is what a continuous solver will be checked against
close to the boundary (fine grids should approach the continuous answer),
not a component it reuses directly. This repository deliberately does not
contain a half-finished continuous solver: that is future work, kept out of
this codebase until it is ready to be done properly rather than bolted onto
the discrete `_QNet` architecture above.

## General-sum — a sanity check, not a second paper

`soccer_nash/markov_game.py` extends pure-first beyond zero-sum: enumerate *all*
stage equilibria (`support_enum.py`), select one by "largest sum of values". On
Battle of the Sexes this correctly avoids the mixed equilibrium whose value is
below either pure one; the soccer game is zero-sum, so it does not touch the
main result. Future work.

## Reproduction

`make dqn` / `make dqn-random` / `make dqn-ablation` regenerate the three
tables above (`nash_dqn.py --seeds 5`, `nash_dqn_random.py --seeds 5`,
`nash_dqn_ablation.py --seeds 2`).
