# The mixed states as geometric templates

`soccer_nash/templates.py`, `scripts/templates.py`.

The decision tree in [geometry.md](geometry.md) *predicts* which stage games are
mixed. This goes one step further: it reduces the 94 no-pure-saddle states of
the 7x5 random-move-order game (gamma 0.9) to a small set of geometric
mechanisms and shows the stage matrix for each.

## Pipeline

```
94 no-pure-saddle states
    -> mirror / role-swap canonicalization      -> 47 pairs
    -> cluster by carrier-frame geometry         -> 8 templates
    -> per template: equilibrium-support subgame,
       matching-pennies test, {U,D,L,R} meaning
```

The mirror map (flip the board left-right, swap the players) is an exact
symmetry of the game, and every mixed state's mirror image is also mixed
(`mirror-closed=True`), so the 94 states are 47 genuine pairs. Clustering the
representatives by the defender's position in the carrier's frame -- forward
lead `dx_rel`, vertical offset `|dy_rel|`, whether the carrier is on a goal
row -- and by the equilibrium support shape gives 8 templates.

## The templates

Matrices are in the carrier's frame: **rows are the carrier's actions
(maximising), columns the defender's (minimising)**, entries are the carrier's
continuation value.

| # | states | geometry | support | structure |
|---|---|---|---|---|
| 1 | 24 | defender 1 ahead, carrier one row off the goal row | 2x2 | matching pennies |
| 2 | 24 | defender 1 ahead, same row, carrier on a goal row | 2x2 | matching pennies |
| 3 | 16 | defender 2 ahead, same row, carrier on a goal row | 2x2 | matching pennies |
| 4 | 12 | defender 1 ahead, same row, carrier on a goal row | 2x1 | defender pure, carrier indifferent |
| 5 | 8 | defender 2 ahead, same row, carrier on a goal row | 3x2 | degenerate (row ties) |
| 6 | 4 | defender 2 ahead, carrier off the goal row | 3x3 | no pure saddle, wider support |
| 7 | 4 | defender diagonally adjacent, carrier off the goal row | 2x2 | matching pennies |
| 8 | 2 | defender 2 ahead, same row, near the back wall | 2x1 | defender pure, carrier indifferent |

**The four 2x2-support templates are all genuine matching pennies** -- best
replies cross in both directions -- and they cover **68 of the 94** mixed
states. Templates 4, 5 and 8 (22 states) are the borderline cases: a pure
saddle is *almost* present (one player has a pure optimal action, the other is
indifferent between two), so they are classed mixed only under the
strict `maximin < minimax` test. Template 6 (4 states) genuinely needs a 3x3
mix.

### Template 2 -- the canonical mechanism

Representative `(2, 1, 3, 1, 1)`: the carrier is on goal row 1, the defender is
one cell directly ahead on the same row.

```
                  defender
                cover-up   drop-back
carrier climb     0.19       0.22
        advance   0.26       0.055
```

- carrier **climb** (step to the next goal row) beats the defender's
  **drop-back**, loses to **cover-up**;
- carrier **advance** (straight at the defender) beats **cover-up** -- the lane
  opens -- and loses to **drop-back**.

Each player's best reply flips with the opponent's choice, so there is no pure
saddle. The equilibrium is `p* = 0.87` climb / `q* = 0.70` cover-up, value
`0.199` -- essentially the kickoff value.

## Why the mix is unavoidable: multi-cell goal + interception geometry

Take a template-2 state: carrier on goal row `r`, defender one cell ahead on row
`r`, both some distance from the goal.

1. **The move order is a coin flip.** Under `move_order="random"` the two chosen
   moves are applied in a random order, each ordering with probability 1/2. So
   when the carrier tries to slip past, whether it or the defender "gets there
   first" is decided by that coin -- neither player controls it.
2. **A multi-cell goal gives the carrier two credible threats.** Rows `r-1`,
   `r`, `r+1` all score. The carrier can *climb* to row `r+1` or *drop* to row
   `r-1` and still be on a scoring lane. The defender, one cell ahead, can cover
   the climb (step up) or the drop (step down) but not both in one move.
3. **Best replies cycle.** If the defender commits to covering the climb, the
   carrier drops and -- half the time, by the move-order coin -- is past before
   the defender recovers; its continuation value goes up. Symmetrically for the
   drop. If the defender holds the cell straight ahead, the carrier climbs or
   drops for free. And for any fixed carrier choice the defender has a strictly
   better cover. No pair of pure actions is a mutual best response.
4. **Therefore the stage game is matching pennies** over
   `{carrier: climb, drop}` x `{defender: cover-up, cover-down}` (or the
   `advance / cover` variant when the defender is not exactly abreast), and its
   value is strictly above the carrier's pure security level.

**Remove any ingredient and the mix disappears:**

- *One-cell goal.* Only row `r` scores. Climbing or dropping takes the carrier
  off the only scoring lane, so "advance / stay on row `r`" is the carrier's
  only useful move and "hold the cell ahead on row `r`" is a pure best reply for
  the defender -- pure saddle. This is why every single-cell-goal board tested
  has zero mixed states ([mechanism.md](mechanism.md)).
- *Deterministic resolution.* The ball carrier always wins a contested cell, so
  "advance into the defender" either clearly works or clearly fails -- there is
  nothing to guess. Every stage game has a pure saddle.
- *Coin-flip tie-break.* A fair coin decides contested cells, but optimal play
  never enters a contest it might lose, so the value equals the deterministic
  one and the pure saddle survives. It is the *move-order* coin, not randomness
  as such, that creates the guess.

So the precise condition for an unavoidable mixed stage game in this model is:

> a stochastic (½–½) resolution order, **and** a goal mouth wider than one cell,
> **and** the defender close enough to contest the carrier's forward cell but
> not able to cover every scoring lane in a single move.

The decision tree's top split -- `defender_can_intercept` -- is exactly the
third clause; templates 1–3 and 7 are exactly the states where all three hold.

## Regenerating

```
python scripts/templates.py --gamma 0.9
```
