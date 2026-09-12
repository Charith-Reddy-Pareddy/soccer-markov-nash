# The mixed states as geometric templates

`soccer_nash/templates.py`, `scripts/templates.py`.

The decision tree in [geometry.md](geometry.md) *predicts* which stage games are
mixed. This goes one step further: it reduces the 94 no-pure-saddle states of
the 7x5 random-move-order game (gamma 0.9) to a small set of geometric
mechanisms and shows the stage matrix for each. **Actions are always `U`, `D`,
`L`, `R`** (up / down / left / right) -- the matrix rows and columns below are
labelled with those four letters only, never a narrative name.

## Pipeline

```
94 no-pure-saddle states
    -> mirror / role-swap canonicalization      -> 47 pairs
    -> cluster by carrier-frame geometry         -> 8 templates
    -> per template: equilibrium-support subgame,
       matching-pennies test, U/D/L/R matrix
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
continuation value, and every row/column is one of `U`, `D`, `L`, `R`.

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
one cell directly ahead on the same row. Carrier's two live actions: `U`, `L`.
Defender's two live actions: `U`, `L`.

```
              defender
             U        L
carrier  U  0.190    0.220
         L  0.260    0.055
```

- carrier `U` (step to the next goal row) beats the defender's `L`, loses to
  the defender's `U`;
- carrier `L` (straight at the defender) beats the defender's `U` -- the lane
  opens -- and loses to the defender's `L`.

Each player's best reply flips with the opponent's choice, so there is no pure
saddle. The equilibrium is `p* = 0.87` on `U` / `q* = 0.70` on `U`, value
`0.199` -- essentially the kickoff value.

### The other three matching-pennies templates

Same reading -- rows the carrier, columns the defender, `[[a,b],[c,d]]` a
crossing game (`a > c` while `b < d`, and `a > b` while `c < d`), so it is
strategically equivalent to matching pennies:

| template | representative | carrier rows | defender cols | matrix | equilibrium |
|---|---|---|---|---|---|
| 1 (24) carrier one row below `r`, defender 1 ahead | `(2,1,3,0,1)` | `D`, `L` | `D`, `R` | `[[0.161, 0.150], [0.022, 0.154]]` | `p*=0.92` on `D`, `q*=0.03` on `D`, value `0.150` |
| 3 (16) defender 2 ahead on `r` | `(1,1,3,1,1)` | `U`, `L` | `U`, `R` | `[[0.205, 0.263], [0.317, 0.049]]` | `p*=0.82` on `U`, `q*=0.66` on `U`, value `0.225` |
| 7 (4) defender diagonally adjacent | `(1,1,1,0,1)` | `L`, `R` | `D`, `L` | `[[0.500, 0.141], [0.181, 0.211]]` | `p*=0.08` on `L`, `q*=0.18` on `D`, value `0.206` |

`scripts/templates.py` prints all eight, including the borderline 2x1 / 3x2
residuals, with the same `U`/`D`/`L`/`R` labelling.

## Why the mix is unavoidable: multi-cell goal + interception geometry

Take a template-2 state: carrier on goal row `r`, defender one cell ahead on row
`r`, both some distance from the goal. The carrier's two live moves are `U`
(step to row `r+1`, still a goal row) and `L` (drive straight at the defender);
the defender's two live moves are `U` (cover row `r+1`) and `L` (hold the cell
directly ahead).

1. **The move order is a coin flip.** Under `move_order="random"` the two chosen
   moves are applied in a random order, each ordering with probability 1/2. So
   when the carrier tries to slip past, whether it or the defender "gets there
   first" is decided by that coin -- neither player controls it.
2. **A multi-cell goal gives the carrier two credible threats.** Rows `r-1`,
   `r`, `r+1` all score, so both `U` (toward `r+1`) and `D` (toward `r-1`) keep
   the carrier on a scoring lane; the defender, one cell ahead, can cover one of
   those rows with its own `U` or `D` but not both in one move.
3. **Best replies cycle.** If the defender commits `U` (covering the row
   above), the carrier plays `D` and -- half the time, by the move-order coin --
   is past before the defender recovers; its continuation value goes up.
   Symmetrically if the defender plays `D`. If the defender instead holds the
   cell straight ahead, the carrier's `U` or `D` is free. And for any fixed
   carrier action the defender has a strictly better reply. No pair of pure
   actions is a mutual best response.
4. **Therefore the stage game is matching pennies** over the carrier's `{U, D}`
   and the defender's `{U, D}` (or `{U, L}` / `{L, R}` in the variants above,
   depending on which two directions are live at that state), and its value is
   strictly above the carrier's pure security level.

**Remove any ingredient and the mix disappears:**

- *One-cell goal.* Only row `r` scores. Playing `U` or `D` takes the carrier
  off the only scoring lane, so staying on row `r` (`L`, straight at the
  defender) is the carrier's only useful move, and holding the cell ahead (`L`)
  is a pure best reply for the defender -- pure saddle. This is why every
  single-cell-goal board tested has zero mixed states ([mechanism.md](mechanism.md)).
- *Deterministic resolution.* The ball carrier always wins a contested cell, so
  playing `L` into the defender either clearly works or clearly fails -- there
  is nothing to guess. Every stage game has a pure saddle.
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
