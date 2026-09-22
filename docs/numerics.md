# Numerical foundations for mixed Nash Q-iteration

Three unsolved problems for the discrete soccer solver, on top of the core
puzzle -- *why does the LP return pure strategies
when Littman's game is supposed to need mixed ones?* `soccer_nash/numerics.py` +
`scripts/numerics.py` address all of them for the zero-sum case, and turn up a
fourth result about the game itself.

Run: `python scripts/numerics.py --move-orders deterministic random --figures`

## Notation: `π1ᵀ Q π2` vs `π2ᵀ Q π1` -- these are not the same quantity

A recurring point of confusion in the group's notes, worth settling once. Fix
`M` = **player 0's** payoff matrix for one stage game: rows are player 0's
actions, columns are player 1's actions, `M[i, j]` is player 0's payoff when
player 0 plays action `i` and player 1 plays action `j`. Player 0 maximises,
player 1 minimises, and (zero-sum) player 1's payoff is `-M[i, j]`.

Let `p` (`= π1`) be player 0's mixed strategy and `q` (`= π2`) be player 1's.
Player 0's expected payoff under `(p, q)` is

    p^T M q          -- rows of M indexed by p, columns by q

**`p^T M q` and `q^T M p` are genuinely different numbers** if you plug them
into the *same* `M` -- `q^T M p` indexes `M`'s rows by `q` (player 1's
strategy) and its columns by `p`, which only makes sense if `M`'s rows are
player 1's actions. Since they are not, `q^T M p` is not player 0's payoff,
not player 1's payoff, not anything meaningful -- it is the classic bug of
swapping which player's strategy goes first without also swapping the matrix
(`soccer_nash/nash_q.py`'s history has exactly this bug, fixed as "Debug
transposed value bracket").

**The safe identity** is the transpose one, which always holds because a
scalar equals its own transpose:

    p^T M q  =  q^T M^T p          -- always true, for any M, p, q

So if your notes write the row player's value as `π2^T Q π1` (column
strategy first), that is consistent with this page's `p^T M q` **only if**
their `Q` is *this page's `M`, transposed* -- i.e. their `Q` has player 1's
actions as rows. Both conventions compute the same number; the rule is
*whichever strategy you write first indexes that matrix's rows*, and you must
transpose to swap the order. `soccer_nash/numerics.py`'s `value_bracket` uses
`p^T M q` throughout (a code comment flags exactly this transpose point at the
one place it matters).

For a **general-sum** game (`soccer_nash/support_enum.py`,
`soccer_nash/markov_game.py`) there is a second, real reason the two players'
values are not simply related: player 1 has its *own* payoff matrix `M2` at
the same `(i, j)` grid, and `M2` is **not** forced to be `-M` the way it is in
a zero-sum game. Player 0's value is `p^T M q`, player 1's is `p^T M2 q` (same
`p`, `q`, same row/column indexing, different matrix) -- and with no relation
between `M` and `M2`, there is no identity connecting the two values at all.
That is the actual content of "general sum", not a notational trap.

## 0. Why the LP returns pure strategies

It is the **collision rule**, not the solver. Under `deterministic` or
`coinflip` resolution the contest outcome is fixed once the joint action is
known, so the carrier has a weakly-dominant reply and every stage game has a
pure saddle -- the LP is *right* to return pure. Littman's random *move order*
makes a ball-steal depend on both players' targets at once, which turns the
contested cell into a matching-pennies game with no pure saddle. Under
`move_order="random"`, 94 of 2380 stage games (γ = 0.9) have no pure saddle.

![Rock-paper-scissors and a soccer stage game side by side; in both, player 0's
best-reply row and player 1's best-reply column never coincide.](figures/png/rps_vs_soccer.png)

The blue bar marks player 0's best row per column, the green bar player 1's best
column per row (an O(A²) check -- no LP needed). In a pure-saddle
game one cell has both; in rock-paper-scissors and in the soccer stage game at
`(0, 1, 1, 1, 1)` -- carrier pinned against its own goal, defender adjacent --
they cycle, and the game has no pure equilibrium.

### 0b. The one exception: tied pure best responses

The paragraph above is not the whole story. Take every state with an exact
pure saddle (`pure_bounds`'s `hi - lo == 0`, checked directly on the raw
payoff matrix, not through the hybrid solver's fast path) and hand its matrix
straight to the same LP (`soccer_nash.matrix_games.solve_zero_sum`) that
Nash-Q calls only for genuinely mixed states -- i.e. ask what the LP itself
would have returned, had nothing short-circuited it. Of the **2,286** such
states on the canonical board, **9** come back with a policy that is *not* a
clean one-hot, even though a pure equilibrium demonstrably exists.

The mechanism is always the same: two of one player's actions are *exactly*
tied against the equilibrium counter-strategy, so the LP has no reason to
prefer either pure vertex and is free to return any point on the tied face --
including a fractional one. State `(0, 0, 2, 1, 0)` is the cleanest example:

```
        U        D        L        R
  U   0.0951   0.0951   0.0848   0.0951
  D   0.0796   0.0764   0.0698   0.0796
  L   0.0796   0.0764   0.0698   0.0796
  R   0.1214   0.0951   0.1214   0.1102
```

Column `D` is the defender's unambiguous pure equilibrium
(`q = [0, 1, 0, 0]`). But against column `D`, rows `U` and `R` **both** pay
exactly `0.0951` -- an exact tie, not an approximate one -- so
`solve_zero_sum` returns `p = [0.719, 0, 0, 0.281]` instead of committing to
either pure row. `certify_game`'s `gap` field still reports `0.0` (it checks
the value bracket in §1 below, not the shape of the returned policy), so the
**certificate** correctly says "this state is pure" even though the
**printed policy**, read on its own, looks like a genuine 72/28 mix.

This is a different phenomenon from Case 9's tie in
[positions.md](positions.md): there, the carrier's own equilibrium mix is
fractional because *it* is indifferent between two actions against a fixed
opponent -- a real mixed equilibrium, just one whose split isn't uniquely
forced. Here, a **pure** equilibrium exists and the fractional-looking output
is purely an artifact of which vertex of the solution polytope the LP happens
to land on. It is rare (9 of 2,286 pure states, **≈0.4%**) but it is real,
reproducible, and exactly the case the certificate is for: check `gap`, not
whether the printed policy happens to look like a one-hot vector.

**Why this never shows up in this project's own results:**
`soccer_nash.nash_q.NashQIteration`'s default `mode="hybrid"` checks
`pure_bounds` *before* ever calling the LP (§0 above), so a state it
classifies as pure never reaches `solve_zero_sum` in the first place -- the
artifact can only appear under `mode="mixed"`, which forces every stage game
through the LP regardless. `scripts/analyze.py` is the one place in this repo
that runs `mode="mixed"`, and it only ever compares *values* between hybrid
and mixed solves, never policies, so the artifact has never leaked into a
reported number -- but reading a `mode="mixed"` policy directly, for any
project that does, is exactly where this would bite.

## 1. The value of a stage game is three numbers

For candidate strategies `(p, q)` the row (maximising) player:

- can **guarantee** `lower = min_j (p M)_j` by committing to `p`;
- **gets** `mid = p^T M q` if both mix;
- can **reach at most** `upper = max_i (M q)_i` if it best-responds to `q`.

At an exact equilibrium all three coincide. `value_bracket` returns all three;
weak duality guarantees the true minimax value lies in `[lower, upper]`, so
`gap = upper - lower` is a **certified** bound on the error of the LP solution
(`certified_value` returns `(mid, gap)`).

| move order | max value-bracket gap | accumulated through Nash Q |
|---|---|---|
| deterministic | `0` | `0` |
| random | `1.1e-16` | duality gap `8.7e-10` |

**With `scipy`'s HiGHS solver the three numbers agree to machine precision**, so
"the first player can do the max thing" -- gaming the gap between `mid` and
`upper` -- buys at most `1.1e-16`. If the gap were real (an older
vertex/simplex solver, or drift accumulated through value iteration), the fix is
to standardise on the **guaranteed value** `lower = min_j (pM)_j`: it is always
a valid lower bound on the true minimax (weak duality), so a player who reports
it is never claiming more than it can defend. Report the pair `[lower, upper]`
as the certificate and `lower` as the single number.

## 2. Rounding to force indifference is scale-blind

Mixed equilibria rely on *exact* indifference, so a stage game that is
essentially rock-paper-scissors plus a tiny epsilon has no exact pure saddle --
but rounding its entries (e.g. to one decimal) manufactures one. Under
discounting the real `gamma^k` value differences are themselves that small.

`classify_stage_game` uses a tolerance that **scales with the matrix entries**
instead of a fixed grid:

- `mixed` -- `minimax - maximin` exceeds the scaled tolerance;
- `pure` -- a *strict* saddle (unique maximin row, unique minimax column);
- `degenerate` -- a saddle exists but is not strict (tied rows/columns).

`rounding_changes_saddle(M, decimals=1)` reports where fixed rounding would flip
the pure-saddle status:

| move order | pure | degenerate | mixed | rounding-to-0.1 flips a saddle |
|---|---|---|---|---|
| deterministic | 0 | 2380 | 0 | **0** |
| random | 604 | 1682 | 94 | **62** |

On the random game, rounding to 0.1 misclassifies **62** stage games -- exactly
the small-entry region where mixed strategies live. The deterministic game is
safe to round (its values are clean `gamma^k` bands well above 0.1) but *every*
one of its 2380 stage games has a **non-strict** saddle: the pure equilibrium
exists everywhere, yet is tie-ridden, which is why the LP sometimes returns a
mix and why the scale-aware classifier is the honest tool.

![Line chart: across every discount from 0.5 to 0.99, fixed 0.1-rounding wrongly
collapses 60-100% of the mixed stage games, and their true minimax-maximin gap
never reaches the 0.1 grain.](figures/png/discounting_trap.png)

**There is no discount at which fixed 0.1-rounding is safe.** At heavy
discounting the genuine `gamma^k` gaps are tiny (median 0.002 at γ = 0.5); at
light discounting the accumulated discount steps leave entries barely apart
(all 102 mixed games flip at γ = 0.99). The scale-aware `classify_stage_game`
tolerance is stable across the whole range: `604 / 1682 / 94` pure /
degenerate / mixed at every `rel_tol` from `1e-12` to `1e-3`.

### 2b. Which of the fourteen documented cases survive rounding?

`rounding_changes_saddle` answers *whether* fixed rounding flips a
classification; `rounding_diagnostic(M)` goes one step further and solves
the *rounded* matrix as its own game, reporting the policy that comes out,
not just a pure/mixed label -- the concrete version of "we definitely need
some kind of approximation rounding" (the project's own Sept 2026 research
meetings). Run against every case in [positions.md](positions.md), at the
full-precision matrix each one's own page actually prints:

| case | what it's for | gap | 3 dec | 2 dec | 1 dec |
|---|---|---|---|---|---|
| 1 | pure, for contrast | 0.0000 | pure | pure | pure |
| 2 | the primary example | 0.0136 | mixed | mixed | **pure** |
| 3 | L/R indifference | 0.0445 | mixed | mixed | **pure** |
| 4 | the corner duel | 0.0121 | mixed | mixed | **pure** |
| 5 | 3-action mix | 0.0047 | mixed | mixed | **pure** |
| 6 | near-pure hedge | 0.0043 | mixed | **pure** | **pure** |
| 7 | mirrored | 0.0136 | mixed | mixed | **pure** |
| 8 | tackle rule | 0.0223 | mixed | mixed | **pure** |
| 9 | asymmetric mix | 0.0095 | mixed | mixed | **pure** |
| 10 | three-lane mix | 0.0699 | mixed | mixed | mixed |
| 11 | reward alone forces the mix | 0.0653 | mixed | mixed | mixed |
| 12 | movement slip | 0.0172 | mixed | mixed | mixed |
| 13 | template 3 | 0.0419 | mixed | mixed | **pure** |
| 14 | template 8 | 0.0178 | mixed | mixed | **pure** |

Bold = a rounding artifact: that precision reports a *different*
equilibrium (usually pure) than the exact solve. Two patterns worth
naming: **Case 6 is the only one that breaks at 2 decimals already** --
consistent with its own page calling it "a near-pure hedge... where
rounding would lie" before this table existed to check it. **Cases 10, 11,
and 12 are the only three that survive even 1-decimal rounding** -- not
coincidentally, `positions.md` already describes Case 10's gap as "the
deepest, most rounding-proof gap of any state on this page." Gap size is
correlated with rounding robustness but does not determine it outright --
Case 12's gap (0.0172) is smaller than Case 8's (0.0223), yet 12 survives
1-decimal rounding and 8 does not, because what actually matters is
whether rounding closes the specific gap between the two matrix entries
the equilibrium sits between, not the gap's raw size.

**Ten of the thirteen genuinely mixed cases collapse to a different,
pure equilibrium at 1-decimal precision.** That is not an argument for
picking 1-decimal rounding *or* against it -- it is why this project's
solver never rounds before solving, and why any rounded diagnostic is
reported as a separate, explicit comparison (as here, and in
[positions.md](positions.md)'s Case 3, which carries this same table
worked out in full as the flagship example) rather than folded into the
reported equilibrium.

## 3. Value iteration vs. freeze-then-iterate

Three schemes were on the table: value iteration (re-solve every stage game
every sweep), the same with a per-sweep Nash cache (`run()` already does this --
it stores `V(s')` once per state and looks it up), and Brandon's
freeze-then-iterate (`run_policy_iteration()` -- solve the stage games, freeze
the strategies, run cheap linear evaluation sweeps, re-sync).

```
random move order, hybrid solver, gamma = 0.9
                  rounds/sweeps   matrix-game solves   wall clock
value iteration       108             9 672              13.8 s
policy iteration       21             1 879              17.2 s
```

`run_policy_iteration()` records the **staleness** Brandon predicted: per outer
round, how far the frozen strategies are from the Nash of the current Q. On the
random game it stays *maximal* (a full pure-strategy flip, `1.0`) for **16
rounds**, then snaps to `< 1e-8` and converges. The freeze trick cuts LP solves
5x but the thrashing phase eats the wall-clock saving; on the deterministic game
it is strictly worse (65 LP solves where value iteration needs 0, because
intermediate value functions have non-strict saddles).

![Line chart: value iteration's Bellman residual climbs to 9 digits of accuracy
over about 100 sweeps.](figures/png/convergence.png)

Value iteration's Bellman residual (`NashQResult.residual_trace`) decays
geometrically at rate ≈ γ. Freeze-then-iterate makes no progress at all until
round 16, because the frozen stage saddle is a pure flip away from the true one
and the linear evaluation sweeps just propagate the wrong continuation values --
until `Q` crosses the threshold where the saddle flips and everything corrects
at once. Concurrent stochastic games have no monotone policy-improvement
guarantee ([discussion.md](discussion.md) §4); this is what that looks like.

### Which value quantity to back up in the frozen sweep

A specific open question: in freeze-then-iterate, the frozen
`(p, q)` do not match the current `Q`, so which of the three stage-game
quantities should the evaluation sweep use? `run_policy_iteration(eval_value=...)`
runs all three (`mid` = `p M q`, `lower` = `min_j (pM)_j`, `upper` =
`max_i (Mq)_i`):

| back up | outer rounds | LP solves | thrash rounds | reaches the fixed point? |
|---|---|---|---|---|
| **`p M q`** | **21** | **1 879** | 16 | yes |
| `min_j (pM)_j` | 42 | 4 100 | 28 | yes |
| `max_i (Mq)_i` | 41 | 4 001 | 28 | yes |

![Line chart of frozen-strategy staleness per outer round for the three choices;
p M q settles to zero by round 16, the two bounds keep spiking until round 28.](figures/png/eval_value.png)

**Back up `p M q`.** All three reach the *same* fixed point -- zero-sum
equilibria are interchangeable, so the frozen strategies always converge to a
Nash and the value with them -- but `p M q` is the *unbiased* estimate while the
strategies are stale, whereas `min_j (pM)_j` is systematically pessimistic and
`max_i (Mq)_i` optimistic. The bias pushes the value function away from the
fixed point and stretches the thrash phase from 16 rounds to 28, doubling the
work.

**Verdict: value iteration with the per-sweep cache wins here.** Freeze-then-
iterate only pays off when the equilibrium solve dominates the backup -- larger
action spaces, or general-sum games.

## 4. Where the game "has to" mix, and what it looks like

On the random-move-order 7x5 board, of the 94 states with no pure saddle
(68 of them a genuinely forced, non-degenerate mix -- [degeneracy.md](degeneracy.md)):

| equilibrium support | count |
|---|---|
| 2x2 (matching pennies) | 68 |
| 2x1 or 1x2 (one side pure) | 14 |
| 3x2 / 2x3 / 3x3 | 12 |

**68 of 94 reduce to a 2x2 matching-pennies mix** (section 0 shows one next to
rock-paper-scissors). The essential sub-game is the carrier choosing between two
of `{U, D, L, R}` and the defender covering one of the same two -- a "guess
where the ball goes" game. [templates.md](templates.md) reduces all 94 to 8
geometric templates and prints the exact `U`/`D`/`L`/`R` stage matrix for each.
