# Player positions and the 4×4 Q matrix, drawn the way the meeting sketched them

At the research meeting the professor drew a stage game by hand: a small grid
of cells, each marked, with arrows chasing each other around the grid to show
that no cell is a stable outcome, at players positioned `(0, 0)` and `(1, 1)`.
`scripts/positions.py` (`make positions`) turns that sketch into a real
diagram, generated from the exact solver instead of drawn free-hand, pairs it
with a picture of where the two players actually are on the board, and
prints the exact **`4x4` `{U, D, L, R}` Q matrix** underneath every case --
always the full grid, never reduced, per the professor's own correction that
the matrix should stay `4x4` and that the matrix itself, not an entropy
number, is the actual output.

**Twelve cases.** Across the whole project there are exactly four
*canonical* equilibrium-support shapes once every stage game is reoriented
so rows are always the carrier's actions -- `(2,2)`: 68 states, `(3,3)`: 4,
`(2,1)`: 14, `(3,2)`: 8. **The defender's support is never larger than the
carrier's**, in any of the 94 mixed states on the canonical board -- a fact
that only became visible once every case in this file was reoriented onto
the same carrier/defender axis. This page shows one example of all four
shapes, plus every distinct *mechanism* studied in the project that can
force a mix: a stochastic move order, this project's own tackle rule, a
dense reward with zero transition noise, and movement slip on a goal shape
that is otherwise always pure.

![Twelve cases, board and Q-matrix graph side by side, including the exact
board position the professor drew.](figures/png/positions.png)

## Reading the graph

Each node is one cell of the complete `4x4` Q matrix (`row = the carrier's
action`, `col = the defender's action`, rows and columns always reoriented so
the carrier's payoff is what's printed, regardless of which player id
actually carries -- see `_oriented_matrix` in `scripts/positions.py`). A blue
arrow points from a cell to the cell in the same column with the higher
payoff -- the carrier's own reason to switch rows. A green arrow points from a
cell to the cell in the same row with the *lower* payoff -- the defender's
reason to switch columns, since the defender minimises. Every case shows the
complete grid, all 16 cells, so the shapes are directly comparable across
cases.

- **A pure state has exactly one node with no outgoing arrow.** That cell is
  the saddle; everything else eventually points to it.
- **A mixed state has arrows everywhere.** Every node points somewhere, so
  the arrows chase each other around the grid with no resting point.

## Case 1 — `(4, 0, 5, 0, 0)`: a pure state, for contrast

Saddle at `U/U`. The carrier plays `U` always; the defender answers `U`
always; neither has any reason to deviate.

```
        U        D        L        R
  U   0.234    0.317    0.523    0.272
  D   0.151    0.211    0.211    0.222
  L   0.161    0.182    0.172    0.190
  R   0.012   -0.095    0.058    0.098
```

## Case 2 — `(0, 1, 1, 1, 0)`: the typical mix

Gap `0.0136`. The carrier mixes `U 63.5% / D 36.5%`; the defender answers
`U 36.5% / R 63.5%`. **This is the common shape**: the carrier is deciding
which goal row to head for, and the defender is guessing which one. 90 of
the 94 mixed states on the canonical board cross a vertical pair exactly
like this one.

```
        U        D        L        R
  U   0.086    0.478    0.286    0.099
  D   0.109    0.076    0.085    0.086
  L   0.103    0.103    0.085    0.084
  R   0.082    0.187   -0.108   -0.071
```

## Case 3 — `(0, 0, 1, 1, 0)`: the corner duel

The board position sketched at the meeting: the carrier at `(0, 0)` -- the
back corner, right against its own goal -- and the defender diagonally
adjacent at `(1, 1)`. This is the closest match, coordinate for coordinate,
to the meeting's own sketch, distinct from the earlier "typical mix" case,
which crosses `U/D`; this one crosses `U/L` for the carrier and `D/L` for
the defender: pinned in the corner, the carrier's two live escapes are
straight up the sideline (`U`) or across along the back line (`L`), and the
defender is guessing which. Gap `0.0121`; the mix is lopsided but genuine
(`U 4.6% / L 95.4%` for the carrier, `D 94.9% / L 5.1%` for the defender) --
the corner leaves little room, but not zero.

```
        U        D        L        R
  U   0.103    0.103   -0.408    0.084
  D   0.109    0.076    0.101    0.086
  L   0.109    0.076    0.101    0.086
  R   0.112   -0.075    0.112    0.088
```

## Case 4 — `(1, 1, 1, 0, 1)`: a clean L/R indifference example

> "when we move right and when we move left, there's an equal chance of me
> winning. That's why I am indifferent between the two."

Gap `0.0445`. The carrier's entire live option set is **exactly `{L, R}`**
-- no vertical option survives at all, which is what makes this state (rather
than Case 3, which crosses `U/L`) the cleanest match to the meeting's *verbal*
description: two actions genuinely equally good against the defender's own
mix (`D 18% / L 82%`). This is the *rarer* shape overall: only 4 of 94 mixed
states cross a purely horizontal pair instead of a vertical one.

```
        U        D        L        R
  U   0.139   -0.173    0.099    0.110
  D   0.171    0.185    0.157    0.171
  L   0.810    0.500    0.141    0.810
  R   0.211    0.181    0.211    0.167
```

## Case 5 — `(0, 0, 2, 0, 0)`: a genuine 3-action mix

Gap `0.0047`, entropy `1.109` bits -- the richest mix on this page. Six cells
from goal, as far as this board allows, with the defender close enough to
threaten three different rows at once: no single row is safely better than
the other two, so three actions are simultaneously undominated. Requested
explicitly at the meeting: the matrix itself, not entropy, is the point.

```
        U        D        L        R
  U   0.084    0.095    0.103    0.110
  D   0.086    0.076    0.076    0.086
  L   0.086    0.076    0.076    0.086
  R   0.088    0.095   -0.088    0.110
```

## Case 6 — `(1, 1, 2, 0, 1)`: a near-pure hedge, where rounding would lie

Gap `0.0043`, entropy only `0.169` bits -- the carrier plays `R 97.5%`,
`D 2.5%`. Reading `0.975` as `1.0` would be the rounding mistake the meeting
flagged: the gap is real and certified, twelve times smaller than a 0.1
rounding grid would resolve.

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

Direct evidence that Case 2's shape is not a one-off coincidence of where it
happens to sit on the board -- the identical mix reappears, exactly
mirrored, wherever the same relative configuration recurs.

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
kind of duel: the carrier mixes `D 60.7% / R 39.3%`, the defender mixes
`U 32.1% / D 67.9%`. Different cause, same structure.

```
        U        D        L        R
  U  -0.012    0.005    0.016   -0.042
  D   0.072   -0.022    0.009    0.000
  L  -0.019   -0.080    0.019   -0.042
  R  -0.007   -0.026   -0.006   -0.039
```

## Case 9 — `(0, 2, 1, 2, 0)`: the asymmetric mix

Support `(2, 1)` -- gap `0.0095`. The odd one out: the carrier's equilibrium
still mixes two actions (`U 2.6% / D 97.4%`), but the **defender's**
equilibrium is a *single fixed move* (`R`, 100%). Every other case on this
page is a symmetric duel -- both players genuinely guessing. Here only one
side is.

```
        U        D        L        R
  U   0.085    0.478    0.291    0.095
  D   0.478    0.085    0.291    0.095
  L   0.093    0.093    0.086    0.093
  R   0.082    0.082   -0.122   -0.072
```

Why the carrier still needs two actions against a defender who never varies:
against the defender's pure `R`, column `R` reads `U 0.095, D 0.095, L 0.093,
R -0.072` -- **`U` and `D` are exactly tied**, both strictly better than `L`
or `R`. Nothing in the payoffs favours one over the other, so any split of
`U`/`D` is an equally valid best reply; the solver's LP reports one particular
split (`2.6% / 97.4%`), not a probability forced by the game the way Case 4's
`18% / 82%` is. This is the meeting's LP-edge-case question made concrete:
**a tie between two rows can look identical, in the printed policy, to a
genuine forced mix, but it is a different phenomenon** -- no best-reply cycle
is involved, and *either* pure `U` or pure `D` alone would also be a valid
equilibrium reply to the defender's fixed `R`. `soccer_nash/certificate.py`'s
`gap` field still distinguishes the two: a tie is a **degenerate** saddle
(row `L`'s pure security value, `0.086`, sits within `0.0095` of the mixed
value), while Case 4's cycle has no pure security value anywhere near it.

## Case 10 — `(0, 2, 2, 2, 1)`: the three-lane mix

Support `(3, 2)` -- gap `0.0699`, the **deepest gap of any state on this
page** (fifteen times deeper than the shallowest hedge in Case 6). This is
the fourth and last canonical support shape: the carrier genuinely needs
three live actions (`U 16.9% / D 66% / L 17.1%`), while the defender only
ever needs two (`L 77.9% / R 22.1%`). Combined with
Case 9, this settles a structural question the certificates make checkable
rather than assumed: **the defender never needs strictly more actions than
the carrier**, across all 94 mixed states on the canonical board.

```
        U        D        L        R
  U   0.247    0.330    0.272    0.317
  D   0.330    0.247    0.272    0.317
  L   0.366    0.366    0.330    0.110
  R   0.228    0.228    0.220    0.205
```

## Case 11 — `(4, 4, 5, 4, 0)`, deterministic: mixing forced by reward alone

Every case so far gets its coin flip from a stochastic *transition* -- the
random move order, or the tackle rule's own coin. This one has **no
transition stochasticity at all** (`move_order="deterministic"`): the mix
comes entirely from `scoring="territory"`, a dense per-step reward for the
ball in the opponent's final third, layered on top of the ordinary win/loss
score. Gap `0.0653`, entropy `0.996` bits -- close to a fair coin, forced
purely by coupling the reward to both players' actions.

```
        U        D        L        R
  U   0.116    0.105    0.170    0.136
  D   0.094    0.415    0.815    0.450
  L   0.094    0.105    0.000    0.105
  R   0.170    0.500   -0.671    0.500
```

## Case 12 — `(1, 3, 1, 4, 1)`: movement slip on a single-cell goal

The sharpest possible contrast with Case 1. A single-cell goal under
deterministic move order is **provably pure at every one of its states** --
that is [the whole goal-width result](result.md). Add `slip=0.15` (each
player independently takes a uniform-random move instead of its chosen one,
15% of the time, regardless of position) to that exact board, and **52 mixed
states appear** where there were 0. Gap `0.0172` -- a genuine, certified
mix, not noise. [generalize.md](generalize.md) states the mechanism
precisely: the goal-width switch is a property of Littman's move-order
coin specifically; a coin that lands on every square regardless of the goal
forces mixing everywhere, independent of goal width.

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
own mix -- that equality is what "indifferent" means in Case 4, and it is
also why the best-response graph above cycles with no resting point: if one
action in the support paid strictly more, that player would raise its weight
on it, which is precisely the condition an equilibrium rules out. A
best-reply cycle and mutual indifference are the same fact seen from two
sides, not two different explanations ([numerics.md](numerics.md) §0). Case 9
shows the boundary of that story: indifference alone (a tie against a fixed
opponent) can produce the same *symptom* -- a fractional policy -- without a
cycle anywhere in the matrix.

## Reproduction

`python scripts/positions.py` prints the exact `4x4` Q matrix and policy for
all twelve states and writes `figures/gallery/positions.svg`. A PDF write-up
of this page is at [positions.pdf](positions.pdf).
