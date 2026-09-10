# Numerical foundations for mixed Nash Q-iteration

The research meeting named three unsolved problems for the discrete soccer
solver, on top of the core puzzle -- *why does the LP return pure strategies
when Littman's game is supposed to need mixed ones?* `soccer_nash/numerics.py` +
`scripts/numerics.py` address all of them for the zero-sum case, and turn up a
fourth result about the game itself.

Run: `python scripts/numerics.py --move-orders deterministic random --figures`

## 0. Why the LP returns pure strategies

It is the **collision rule**, not the solver. Under `deterministic` or
`coinflip` resolution the contest outcome is fixed once the joint action is
known, so the carrier has a weakly-dominant reply and every stage game has a
pure saddle -- the LP is *right* to return pure. Littman's random *move order*
makes a ball-steal depend on both players' targets at once, which turns the
contested cell into a matching-pennies game with no pure saddle. Under
`move_order="random"`, 94 of 2380 stage games (γ = 0.9) are genuinely mixed.

![Rock-paper-scissors and a soccer stage game side by side; in both, player 0's
best-reply row and player 1's best-reply column never coincide.](figures/png/rps_vs_soccer.png)

The blue bar marks player 0's best row per column, the green bar player 1's best
column per row (the professor's O(A²) check -- no LP needed). In a pure-saddle
game one cell has both; in rock-paper-scissors and in the soccer stage game at
`(0, 1, 1, 1, 1)` -- carrier pinned against its own goal, defender adjacent --
they cycle, and the game has no pure equilibrium.

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

The professor's specific open question: in freeze-then-iterate, the frozen
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

On the random-move-order 7x5 board, of the 94 states that require mixed
strategies:

| equilibrium support | count |
|---|---|
| 2x2 (matching pennies) | 68 |
| 2x1 or 1x2 (one side pure) | 14 |
| 3x2 / 2x3 / 3x3 | 12 |

**68 of 94 reduce to a 2x2 matching-pennies mix** (section 0 shows one next to
rock-paper-scissors). The essential sub-game is the carrier choosing between
{advance toward goal, climb / hold} and the defender between {cover a lane, hold
the forward cell} -- a "guess where the ball goes" game. [templates.md](templates.md)
reduces all 94 to 8 geometric templates and prints a stage matrix for each.
