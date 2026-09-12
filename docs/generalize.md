# How far the goal-width switch generalizes, and why the mixed states arise

`scripts/generalize.py` (`make generalize`), `experiments/generalize.csv`.

The [result.md](result.md) claim is that under Littman's random move order a
single goal cell keeps every stage game pure while a wider goal forces mixed
stage games. This page pushes on that claim from three sides.

## A. Board scale and goal shape

The switch survives larger boards and odd goal shapes. Sweeping the random
move order over boards up to `11x5` and `9x7` (about 7 800 states) and over
goals that sit against a wall or skip a row:

- **One goal cell: 0 mixed stage games, every configuration.**
- **Two or more goal cells: mixed stage games appear, every configuration**, at
  roughly 3 to 5 percent of the reachable states.
- **Adjacency does not matter.** `goal_rows = (1, 3)` with a gap at row 2, or
  `(0, 2, 4)` with two gaps, still produce mixed stage games. What counts is
  that the carrier has two or more scoring cells the defender cannot cover at
  once, not that the cells touch.

So within the random-move-order family the switch is robust to board size,
board aspect ratio, and goal placement.

## B. A different kind of stochasticity breaks the switch

`SoccerGame(slip=p)` adds noise that does not depend on either player's choice:
each turn, each player independently takes a uniform-random move instead of its
chosen one with probability `p`. This is layered on top of `move_order`.

With `slip > 0` the game has mixed stage games **even under the deterministic
move order and even with a single goal cell**. On a `5x5` single-cell board that
is pure everywhere at `slip = 0`, deterministic resolution with `slip = 0.1`
gives 62 genuinely mixed stage games (mixing entropy median around 0.7 bits, so
not hairline cases), and the count stays near that through `slip = 0.3`.

The reading: the goal-width switch is a statement about **Littman's move-order
rule**, not about stochastic transitions in general. What forces a stage game
to mix is a transition outcome that turns on a coin neither player controls.
The random move order supplies that coin only where the carrier has two lanes
and the defender is between them, so mixing tracks the goal width. Movement
slip supplies the coin on every square, so mixing appears regardless of the
goal. A fair coin flip on contested squares (`move_order="coinflip"`) supplies
no such coin on the equilibrium path, because optimal play simply avoids the
contested square, which is why that variant stays pure at every goal width
([blend.md](blend.md), [numerics.md](numerics.md)).

## C. What the mixed stage games are, concretely

For the canonical `7x5` three-row-goal board, every mixed stage game contains a
`2x2` matching-pennies submatrix ([certificate.py](../soccer_nash/certificate.py)).
Reading which two carrier moves and which two defender moves cross in that
submatrix:

- The carrier's crossing pair includes a vertical move (up or down) in 90 of the
  94 mixed states. The carrier is choosing **which goal row to head for**.
- The defender's crossing pair mirrors it. The defender is **guessing which row
  the carrier will take**.
- Pure `L`/`R` crossings (no vertical choice at all) occur in only 4 states.

This is the mechanism in one line: near a multi-cell goal the carrier picks a
lane and the defender guesses the lane, which is matching pennies, so neither
side has a safe pure move.

## Where this leaves the generalization question

Confirmed to generalize: across board size, board shape, and goal placement,
for the random move order.

Confirmed **not** to generalize: across the transition rule. Action-independent
movement noise produces mixed stage games at any goal width, so the clean
one-cell-versus-wider switch is specific to the move-order mechanism.

Still open: a board-size-free proof of the one-cell pure result
([proof.md](proof.md)); an exact geometric predicate for the mixed set (the
carrier-frame decision tree reaches precision 0.96, recall 0.91); and whether
the switch has an analogue in the general-sum or many-player versions.
