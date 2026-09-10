# RQ3: where does the game *have* to mix?

`scripts/geometry_model.py`, `scripts/phase_diagram.py`.

## The mechanism: a multi-cell goal mouth

Earlier notes attributed the mixed-strategy states to "large boards". The phase
diagram (`experiments/phase_diagram.csv`, `figures/png/phase_diagram.png`) sweeps 46
board / goal-width configurations and shows the real gate:

| goal-mouth width | configs | mixed states |
|---|---|---|
| 1 | 17 (boards 3x3 .. 9x5) | **0 in every one** |
| >= 2 | 29 | 3-12% of states, in every one |

The mixed fraction drifts *down* with board area (more midfield filler) and is
roughly flat in goal width beyond 2 -- so it is a threshold in goal width, not a
board-size effect.

**A state requires mixed strategies only when the goal is more than one cell
wide.** With a one-cell goal the carrier's only winning approach is that single
cell, so the defender knows exactly where to stand and every stage game has a
pure saddle. With a wider goal the carrier can threaten more than one cell at
once, and under the random move order the defender cannot cover all of them --
it has to *guess*, which is a matching-pennies subgame ([templates.md](templates.md)).

## Geometry predicts the classification

The decision tree here is a *discovery tool*: it points at the geometric
condition, which [templates.md](templates.md) and [proof.md](proof.md) then turn
into an explicit mechanism and a partial theorem. It is not the end of the
analysis.

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

The 94 mixed states (7x5, gamma 0.9) canonicalize under the board mirror to 47
pairs, then cluster into **8 geometric templates**. The four with a 2x2
equilibrium support are all verified matching pennies (best replies cross both
ways) and cover **68 of 94** states; the rest are borderline near-pure saddles
or a single 3x3 mix. Each template's stage matrix, action meanings, and the
analytic argument for why the mix is unavoidable are in
[templates.md](templates.md).
