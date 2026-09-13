# Player positions and the 4×4 Q matrix

We examine twelve representative soccer states. For each, the **left panel**
shows the physical player configuration and every action either player
actually takes in equilibrium, drawn as probability arrows (thick = likely,
thin = unlikely, no arrow at all = zero probability). The **right panel**
shows the complete **`4x4` stage-game Q matrix** -- the payoff table
`Q(s, a0, a1)` at that one state, not the value function `V(s)` -- redrawn as
a node-and-arrow graph, one node per cell, an arrow toward whichever cell
either player would rather deviate to. Always the full grid, never reduced.
The goal is not simply to show *where* mixing happens on the board, but
**what the players are actually doing when it does**.

A player's **support** is the set of actions it assigns positive probability
in an equilibrium strategy -- support `(2,2)` means the carrier mixes over 2
actions and the defender mixes over 2 actions.

**Canonical support shapes.** Across the whole project there are exactly
four of them, once every stage game is reoriented so rows are always the
carrier's actions:

| Carrier support | Defender support | Shape | States |
|:---:|:---:|:---:|---:|
| 2 actions | 2 actions | `(2,2)` | 68 |
| 3 actions | 3 actions | `(3,3)` | 4 |
| 2 actions | 1 action | `(2,1)` | 14 |
| 3 actions | 2 actions | `(3,2)` | 8 |
| | | **total** | **94** |

**The defender's support is never larger than the carrier's**, in any of the
94 mixed states on the canonical board. The bigger finding underneath those
four rows: **94 mixed states are not 94 different phenomena -- they are four
repeated shapes, recurring all over the board.** Case 7 shows one of those
repeats directly (Case 2's exact shape, elsewhere on the board, mirrored).
This page shows one example of each of the four shapes, plus every distinct
*mechanism* studied in the project that can force a mix: a stochastic move
order, this project's own tackle rule, a dense reward with zero transition
noise, and movement slip on a goal shape that is otherwise always pure.

![Twelve cases, board and Q-matrix graph side by side.](figures/png/positions.png)

## Reading the board and the Q matrix

**The board (left).** Each player is a filled circle; the ball sits on
whoever carries it. An arrow points from a player toward a cell it might
move to, and its **thickness and opacity are the equilibrium probability**
of that action -- a bold arrow is very likely, a thin one is unlikely, and
an action with (numerically) zero probability draws no arrow at all. A pure
policy is one bold arrow and nothing else; a mixed policy fans out into two
or three arrows of different weights. Percentages are labelled next to every
arrow that isn't at 100%.

**The Q matrix (right).** Each node is one cell of the complete `4x4` stage
matrix (`row = the carrier's action`, `col = the defender's action`, always
reoriented so the printed payoff is the carrier's, regardless of which
player id actually carries -- see `_oriented_matrix` in
`scripts/positions.py`). A blue arrow points from a cell to the cell in the
same column with the higher payoff -- the carrier's own reason to switch
rows. A green arrow points from a cell to the cell in the same row with the
*lower* payoff -- the defender's reason to switch columns, since the
defender minimises. Every case shows the complete grid, all 16 cells, so the
shapes are directly comparable across cases. This is the mathematical
justification *for* what the board already shows: it is why the players'
support is exactly what it is.

- **A pure state has exactly one node with no outgoing arrow.** That cell is
  the saddle; everything else eventually points to it.
- **A mixed state has arrows everywhere.** Every node points somewhere, so
  they chase each other around a closed best-response cycle: no pure action
  pair is stable, which is exactly why the board shows more than one arrow.

## Case 1 — `(4, 0, 5, 0, 0)`: a pure state, for contrast

Saddle at `U/U`. The carrier plays `U` always; the defender answers `U`
always; neither has any reason to deviate.

**Support:** carrier `{U}` (100%) · defender `{U}` (100%).

![Case 1 board position and Q matrix graph.](figures/png/positions_case01.png)

```
        U        D        L        R
  U   0.234    0.317    0.523    0.272
  D   0.151    0.211    0.211    0.222
  L   0.161    0.182    0.172    0.190
  R   0.012   -0.095    0.058    0.098
```

## Case 2 — `(0, 1, 1, 1, 0)`: the primary example -- the typical mix

**This is the case to lead with.** Gap `0.0136`. The carrier mixes
`U 63.5% / D 36.5%`; the defender answers `U 36.5% / R 63.5%`. This is not
one strange state among many -- it is the dominant pattern in the entire
mixed-state set: **90 of the 94 mixed states on the canonical board** cross
a vertical pair exactly like this one, the carrier deciding which goal row
to attack and the defender guessing which one. Everything that follows is
either this same shape recurring (Case 7), or a genuinely different shape
worth contrasting against it (Cases 3, 4, 5, 9, 10).

**Support:** carrier `{U, D}` (63.5% / 36.5%) · defender `{U, R}`
(36.5% / 63.5%).

![Case 2 board position and Q matrix graph.](figures/png/positions_case02.png)

```
        U        D        L        R
  U   0.086    0.478    0.286    0.099
  D   0.109    0.076    0.085    0.086
  L   0.103    0.103    0.085    0.084
  R   0.082    0.187   -0.108   -0.071
```

## Case 3 — `(1, 1, 1, 0, 1)`: a clean L/R indifference example

Placed directly after the primary example on purpose: this is the cleanest
demonstration of the intuition Case 2 only implies --

> "when we move right and when we move left, there's an equal chance of me
> winning. That's why I am indifferent between the two."

Gap `0.0445`. The carrier's entire live option set is **exactly `{L, R}`**
-- no vertical option survives at all, which is what makes this state (rather
than Case 4, which crosses `U/L`) the cleanest match to that description:
two actions genuinely equally good against the defender's own mix. This is
the *rarer* shape overall: only 4 of 94 mixed states cross a purely
horizontal pair instead of a vertical one.

**Support:** carrier `{L, R}` (7.7% / 92.3%) · defender `{D, L}`
(18% / 82%).

![Case 3 board position and Q matrix graph.](figures/png/positions_case03.png)

```
        U        D        L        R
  U   0.139   -0.173    0.099    0.110
  D   0.171    0.185    0.157    0.171
  L   0.810    0.500    0.141    0.810
  R   0.211    0.181    0.211    0.167
```

## Case 4 — `(0, 0, 1, 1, 0)`: the corner duel

The board position sketched at the meeting: the carrier at `(0, 0)` -- the
back corner, right against its own goal -- and the defender diagonally
adjacent at `(1, 1)`. This is the closest match, coordinate for coordinate,
to that sketch. Distinct from both cases before it -- Case 2 crosses `U/D`,
Case 3 a clean `L/R` -- this one crosses `U/L` for the carrier and `D/L` for
the defender: pinned in the corner, the carrier's two live escapes are
straight up the sideline (`U`) or across along the back line (`L`), and the
defender is guessing which. Gap `0.0121`; the mix is lopsided but genuine --
the corner leaves little room, but not zero.

**Support:** carrier `{U, L}` (4.6% / 95.4%) · defender `{D, L}`
(94.9% / 5.1%).

![Case 4 board position and Q matrix graph.](figures/png/positions_case04.png)

```
        U        D        L        R
  U   0.103    0.103   -0.408    0.084
  D   0.109    0.076    0.101    0.086
  L   0.109    0.076    0.101    0.086
  R   0.112   -0.075    0.112    0.088
```

## Case 5 — `(0, 0, 2, 0, 0)`: a genuine 3-action mix

Six cells from goal, as far as this board allows, with the defender close
enough to threaten three different rows at once: no single row is safely
better than the other two, so three actions are simultaneously undominated
on both sides. This is one of only **4 states on the whole canonical board**
with support shape `(3,3)` -- mixing is not always a 50/50 split between two
actions. (Entropy 1.109 bits, the richest mix on this page, if a single
summary number is wanted -- but the three-way split itself is the point.)

**Support:** carrier `{U, L, R}` (43.3% / 54.7% / 1.9%) · defender
`{U, D, L}` (91.9% / 5.9% / 2.3%).

![Case 5 board position and Q matrix graph.](figures/png/positions_case05.png)

```
        U        D        L        R
  U   0.084    0.095    0.103    0.110
  D   0.086    0.076    0.076    0.086
  L   0.086    0.076    0.076    0.086
  R   0.088    0.095   -0.088    0.110
```

## Case 6 — `(1, 1, 2, 0, 1)`: a near-pure hedge, where rounding would lie

Gap `0.0043`. The **defender** plays `R 97.5% / D 2.5%` -- entropy only
0.169 bits, a hedge so lopsided that reading `0.975` as `1.0` would be a
real rounding mistake: the gap is certified and real, twelve times smaller
than a 0.1 rounding grid would resolve. The carrier's own mix is more
balanced (`D 73.9% / L 26.1%`) -- it is the defender's near-pure hedge that
makes this case worth including, not the carrier's.

**Support:** carrier `{D, L}` (73.9% / 26.1%) · defender `{D, R}`
(2.5% / 97.5%).

![Case 6 board position and Q matrix graph.](figures/png/positions_case06.png)

```
        U        D        L        R
  U   0.317    0.317    0.247   -0.178
  D   0.211    0.211    0.211    0.167
  L   0.171    0.045    0.157    0.171
  R   0.182    0.182    0.182    0.135
```

## Case 7 — `(5, 1, 6, 1, 1)`: the typical mix, mirrored

`soccer_nash.symmetry.mirror_state` flips the board left-right and swaps
which player carries; this is Case 2's exact image under that map. The
values negate to machine precision:

    V(case 2) = +0.094235
    V(mirror) = -0.094235
    sum       = -1.39e-17   (zero, to floating-point noise)

Direct evidence for the repetition claim above: this is not a new shape, it
is Case 2's identical mix reappearing, exactly mirrored, wherever the same
relative configuration recurs on the board.

**Support:** carrier `{U, D}` (63.5% / 36.5%) · defender `{U, L}`
(36.5% / 63.5%).

![Case 7 board position and Q matrix graph.](figures/png/positions_case07.png)

```
        U        D        L        R
  U   0.086    0.478    0.099    0.286
  D   0.109    0.076    0.086    0.085
  L   0.082    0.187   -0.071   -0.108
  R   0.103    0.103    0.084    0.085
```

## Case 8 — `(2, 3, 3, 3, 1)`, on a 5×4 board: this project's own tackle rule

Every other case here gets its coin flip from Littman's random move order.
This one is solved under [the tackle rule](tackle.md)
(`move_order="tackle", tackle_prob=0.5`) instead -- the defender commits to a
challenge that wins the ball with some fixed probability or bounces off, a
mechanism with nothing to do with move order at all. It produces the same
kind of duel (gap 0.0223): the carrier mixes `U 32.1% / D 67.9%`, the
defender mixes `D 60.7% / R 39.3%`. Different cause, same structure.

**Support:** carrier `{U, D}` (32.1% / 67.9%) · defender `{D, R}`
(60.7% / 39.3%).

![Case 8 board position and Q matrix graph.](figures/png/positions_case08.png)

```
        U        D        L        R
  U  -0.012    0.005    0.016   -0.042
  D   0.072   -0.022    0.009    0.000
  L  -0.019   -0.080    0.019   -0.042
  R  -0.007   -0.026   -0.006   -0.039
```

## Case 9 — `(0, 2, 1, 2, 0)`: the asymmetric mix -- the case worth discussing most

Support `(2, 1)` -- gap `0.0095`. The odd one out: the carrier's equilibrium
still mixes two actions (`U 2.6% / D 97.4%`), but the **defender's**
equilibrium is a *single fixed move* (`R`, 100%). Every other case on this
page is a symmetric duel -- both players genuinely guessing. Here only one
side is.

**Support:** carrier `{U, D}` (2.6% / 97.4%) · defender `{R}` (100%).

![Case 9 board position and Q matrix graph.](figures/png/positions_case09.png)

```
        U        D        L        R
  U   0.085    0.478    0.291    0.095
  D   0.478    0.085    0.291    0.095
  L   0.093    0.093    0.086    0.093
  R   0.082    0.082   -0.122   -0.072
```

Why the carrier still needs two actions against a defender who never
varies: against the defender's pure `R`, column `R` reads `U 0.095, D 0.095,
L 0.093, R -0.072` -- **`U` and `D` are exactly tied**, both strictly better
than `L` or `R`. Nothing in the payoffs favours one over the other, so any
split of `U`/`D` is an equally valid best reply; the solver's LP reports one
particular split (`2.6% / 97.4%`), not a probability forced by the game the
way Case 3's `7.7% / 92.3%` is. This is the sharpest example on this page of
a general fact worth stating plainly: **a fractional LP output is not
automatically a strategically required mixed strategy.** A tie between two
rows can look identical, in the printed policy, to a genuine forced mix, but
it is a different phenomenon -- no best-reply cycle is involved, and *either*
pure `U` or pure `D` alone would also be a valid equilibrium reply to the
defender's fixed `R`. `soccer_nash/certificate.py`'s `gap` field still
distinguishes the two: a tie is a **degenerate** saddle (row `L`'s pure
security value, `0.086`, sits within `0.0095` of the mixed value), while
Case 3's cycle has no pure security value anywhere near it.

## Case 10 — `(0, 2, 2, 2, 1)`: the three-lane mix

Support `(3, 2)` -- gap `0.0699`, the **deepest gap of any state on this
page** (fifteen times deeper than the shallowest hedge in Case 6). This is
the fourth and last canonical support shape: the carrier genuinely needs
three live actions, while the defender only ever needs two. Combined with
Case 9, this settles a structural question the certificates make checkable
rather than assumed: **the defender never needs strictly more actions than
the carrier**, across all 94 mixed states on the canonical board.

**Support:** carrier `{U, D, L}` (16.9% / 66% / 17.1%) · defender `{L, R}`
(77.9% / 22.1%).

![Case 10 board position and Q matrix graph.](figures/png/positions_case10.png)

```
        U        D        L        R
  U   0.247    0.330    0.272    0.317
  D   0.330    0.247    0.272    0.317
  L   0.366    0.366    0.330    0.110
  R   0.228    0.228    0.220    0.205
```

## Case 11 — `(4, 4, 5, 4, 0)`, deterministic: mixing forced by reward alone

An advanced case, deliberately placed after the ten core examples above.
Every case so far gets its coin flip from a stochastic *transition* -- the
random move order, or the tackle rule's own coin. This one has **no
transition stochasticity at all** (`move_order="deterministic"`): the mix
comes entirely from `scoring="territory"`, a dense per-step reward for the
ball in the opponent's final third, layered on top of the ordinary win/loss
score. Gap `0.0653` -- close to a fair coin (entropy 0.996 bits), forced
purely by coupling the reward to both players' actions. **Mixing does not
require stochastic transitions; it can come from the payoff structure
alone.**

**Support:** carrier `{D, R}` (53.8% / 46.2%) · defender `{U, L}`
(95.1% / 4.9%).

![Case 11 board position and Q matrix graph.](figures/png/positions_case11.png)

```
        U        D        L        R
  U   0.116    0.105    0.170    0.136
  D   0.094    0.415    0.815    0.450
  L   0.094    0.105    0.000    0.105
  R   0.170    0.500   -0.671    0.500
```

## Case 12 — `(1, 3, 1, 4, 1)`: movement slip on a single-cell goal

The sharpest possible contrast with Case 1. A single-cell goal under
deterministic move order is **pure across every configuration tested in
this project** -- that is [the whole goal-width result](result.md); a
board-size-free proof remains open (see [generalize.md](generalize.md)).
Add `slip=0.15` (each player independently takes a uniform-random move
instead of its chosen one, 15% of the time, regardless of position) to that
exact board, and **52 mixed states appear** where there were 0. Gap `0.0172`
-- a genuine, certified mix, not noise. [generalize.md](generalize.md)
states the mechanism precisely: the goal-width switch is a property of
Littman's move-order coin specifically; a coin that lands on every square
regardless of the goal forces mixing everywhere, independent of goal width.

**Support:** carrier `{D, L}` (80.2% / 19.8%) · defender `{U, L}`
(3.9% / 96.1%).

![Case 12 board position and Q matrix graph.](figures/png/positions_case12.png)

```
        U        D        L        R
  U  -0.273    0.006    0.001    0.069
  D  -0.031    0.069    0.057    0.450
  L   0.396    0.083    0.039    0.450
  R   0.039    0.020    0.015    0.092
```

## The mathematics behind all twelve

A mixed equilibrium is exactly the strategy pair where every action in a
player's support earns the **same expected payoff** against the opponent's
own mix -- that equality is what "indifferent" means in Case 3, and it is
also why the best-response graph above cycles with no resting point: if one
action in the support paid strictly more, that player would raise its weight
on it, which is precisely the condition an equilibrium rules out. A
best-reply cycle and mutual indifference are the same fact seen from two
sides, not two different explanations ([numerics.md](numerics.md) §0). Case 9
shows the boundary of that story: indifference alone (a tie against a fixed
opponent) can produce the same *symptom* -- a fractional policy -- without a
cycle anywhere in the matrix.

## Reproduction

`python scripts/positions.py` prints the exact `4x4` Q matrix, policy, and
action support for all twelve states and writes
`figures/gallery/positions.svg`. A PDF write-up of this page is at
[positions.pdf](positions.pdf).
