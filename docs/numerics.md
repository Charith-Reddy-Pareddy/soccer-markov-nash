# Numerical foundations for mixed Nash Q-iteration

The research meeting named three unsolved problems for the discrete soccer
solver. `soccer_nash/numerics.py` + `scripts/numerics.py` address all three for
the zero-sum case, and turn up a fourth result about the game itself.

Run: `python scripts/numerics.py --move-orders deterministic random`

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

**With `scipy`'s HiGHS solver the three numbers agree to machine precision.**
The concern ("the first player can do the max thing") is real for older
vertex/simplex solvers but not a practical problem here -- the LP solution is a
certified `1e-16`-equilibrium at every stage.

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

**68 of 94 reduce to a 2x2 matching-pennies mix.** The essential sub-game is the
carrier choosing between {advance toward goal, hold / contest} and the defender
choosing between {block, intercept} -- a "guess where the ball goes" game. That
is why Littman's *random move order* creates the mixing and the deterministic
A10 rule does not: the move-order coin makes the contested-square outcome depend
on both players' hidden action choices at once.
