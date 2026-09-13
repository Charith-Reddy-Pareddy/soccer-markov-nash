# The best-response-vs-greedy deep dive

[tournament.md](tournament.md), including its "plain 4×4" section, already
shows *that* minimax survives a tailored challenger while greedy collapses
(`+0.13` vs `−0.54` goal difference, on the plain 4×4 board) and assert, in
one closing paragraph, *why*: greedy commits to a single row at the mixed
states, and the challenger punishes it there. This page tests that claim
directly instead of leaving it as an assertion -- `scripts/tournament_deepdive.py`,
run once, ~80 s.

## 1. The patched-policy test

Take greedy/rand's policy and patch it at **only** the states this board's
exact solve classifies as genuinely mixed (no pure saddle) -- play the exact
Nash mix there, leave every other state exactly as greedy already plays it --
then build a **fresh** challenger for that patched policy (not greedy's own
challenger; a challenger specifically built to beat the patched policy) and
re-score it.

| | vs. its own challenger |
|---|---|
| greedy/rand | `−0.535` |
| greedy/rand, patched at the 56/760 mixed states only | `−0.252` |
| minimax | `+0.130` |

Patching **7.4% of the state space** (56 of 760 states) recovers **43%** of
the gap between greedy and minimax against a tailored opponent. That is a
real, substantial, *partial* effect -- not the whole story (patching does not
reach minimax's robustness), but confirmation that a meaningful share of
minimax's advantage is concentrated exactly where the theory says it should
be: the states that need a genuine mix, not the pure ones a deterministic
policy already gets right.

## 2. One concrete exploited state

The single highest-occupancy mixed state under (greedy, its own challenger)
where greedy's fixed action differs from what minimax plays there:

State `(1, 2, 3, 2, 0)` -- occupancy `0.154` under (greedy, challenger), i.e.
roughly one visit in six over the whole discounted trajectory from kickoff.

```
        U        D        L        R
  U   0.117    0.129    0.082    0.117
  D   0.249    0.143    0.225    0.154
  L   0.117    0.129    0.118    0.117
  R   0.277    0.277   -0.000    0.171
```

Greedy/rand plays `R` here, always. Against that fixed row, the challenger's
best reply is `L` -- payoff `-0.000`, the worst column for row `R` (compare
`+0.277` for `U` or `D` against the same column). Minimax's exact mix at this
state is `{D: 0.707, R: 0.293}`: no single row for the challenger to key on,
so its best reply is indifferent between columns and can't drive the value
down the way it does against greedy's single committed row.

## 3. Gamma sweep: does the recovered share hold?

Re-running the patched-policy test at six discount factors turns up a
genuine, non-obvious trend -- the recovered share is not constant, it falls
steadily as the horizon lengthens, and **reverses sign** at `gamma=0.99`:

| gamma | minimax | greedy/rand | patched | mixed states | % of gap recovered |
|---|---|---|---|---|---|
| 0.50 | `+0.006` | `−0.016` | `−0.003` | 56/760 (7.4%) | 60.2% |
| 0.70 | `+0.038` | `−0.089` | `−0.017` | 56/760 (7.4%) | 57.0% |
| 0.80 | `+0.075` | `−0.189` | `−0.057` | 56/760 (7.4%) | 50.1% |
| 0.90 | `+0.130` | `−0.535` | `−0.252` | 56/760 (7.4%) | 42.6% |
| 0.95 | `+0.167` | `−1.367` | `−0.775` | 64/760 (8.4%) | 38.6% |
| 0.99 | `+0.213` | `−4.879` | `−5.057` | 60/760 (7.9%) | **−3.5%** |

Two things hold at every discount: **minimax stays positive** against its
challenger throughout (never exploitable, the headline claim of
tournament.md/tournament4.md is not a `gamma=0.9` artifact), and **greedy
gets catastrophically worse** as gamma grows (`−0.02` at `gamma=0.5` to
`−4.88` at `gamma=0.99` -- a longer horizon gives the challenger more
discounted turns to punish the same committed rows). What is new here: the
*patch* helps less and less as gamma grows, and at `gamma=0.99` it is
slightly **worse** than doing nothing. Patching only the locally-mixed states
leaves a policy that is sometimes minimax, sometimes greedy -- consistent at
each state in isolation, but the seam between the two regimes is itself
something a sufficiently long-horizon challenger can apparently route
around and exploit, in a way neither pure policy alone offers it. The
mixed-state patch is a real, substantial contributor to robustness at
ordinary discounting, not a complete explanation, and its share of the
credit shrinks (and briefly inverts) as the horizon gets very long.

## Reproduction

`python scripts/tournament_deepdive.py` prints all three sections and writes
`experiments/tournament_deepdive.csv` (the gamma sweep). Uses the same plain
4×4 board as tournament.md's "plain 4×4" section (5×4, two-cell goals, `{U, D,
L, R}`, random move order).
