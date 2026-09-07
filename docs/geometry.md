# RQ3: where does the game *have* to mix?

`scripts/geometry_model.py`, `scripts/experiments.py board|goalmouth`.

## The mechanism: a multi-cell goal mouth

Earlier notes attributed the mixed-strategy states to "large boards". The board
sweep (`experiments/board_sweep.csv`) shows the real gate:

| board height | goal rows | mixed states |
|---|---|---|
| 3 (any width 3-11) | 1 | **0** |
| >= 4 | >= 2 | grows with board area |

And directly (`experiments/goal_mouth_sweep.csv`, fixed 5x9 board):

| goal rows | mixed states |
|---|---|
| 1 | 0 |
| 2 | (grows) |
| ... | ... |

**A state requires mixed strategies only when the goal is more than one cell
wide.** With a one-cell goal the carrier's only winning approach is that single
cell, so the defender knows exactly where to stand and every stage game has a
pure saddle. With a wider goal the carrier can threaten more than one cell at
once, and under the random move order the defender cannot cover all of them --
it has to *guess*, which is a matching-pennies subgame.

## Geometry predicts the classification

Features are computed in the carrier's frame (invariant under the board mirror;
`soccer_nash/geometry.py`). For the 7x5 random game:

| feature | P(mixed &#124; feature) | P(mixed &#124; not) |
|---|---|---|
| `defender_can_intercept` (defender within one move of the carrier's forward cell) | **0.44** | 0.002 |
| `defender_ahead_same_row` | 0.60 | 0.025 |
| `adjacent` | 0.17 | 0.025 |
| `carrier_can_score_next` | 0.00 | 0.043 |

A depth-4 decision tree (`soccer_nash/tree.py`) predicts "mixed" with
**precision 0.96, recall 0.91** from geometry alone. The rule reduces to:

> A stage game needs mixed strategies almost exactly when the defender is within
> one move of intercepting the carrier's forward cell *and* is lined up with a
> goal-row threat -- either the carrier is on a goal row with the defender
> directly ahead, or the carrier is one row off with the defender between it and
> the goal.

## Templates

Of the 94 mixed states (7x5, gamma 0.9), by (equilibrium support, defender
geometry):

| support | defender geometry | count |
|---|---|---|
| 2x2 | ahead of the carrier | 40 |
| 2x2 | directly ahead, same row | 24 |
| 2x1 / 1x2 | directly ahead, same row | 12 |
| 3x3 / 2x3 / 3x2 | ahead | 12 |
| 2x2 | merely adjacent | 4 |
| 2x1 / 1x2 | ahead | 2 |

**~88 of 94 have the defender ahead of the carrier**; ~68 are a 2x2
matching-pennies mix of carrier {advance, hold} against defender {block,
intercept}.
