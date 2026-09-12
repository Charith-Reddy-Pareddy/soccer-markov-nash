# Player positions and the 4×4 Q matrix, drawn the way the meeting sketched them

At the research meeting the professor drew a stage game by hand: a small grid
of cells, each marked, with arrows chasing each other around the grid to show
that no cell is a stable outcome. `scripts/positions.py` (`make positions`)
turns that sketch into a real diagram, generated from the exact solver instead
of drawn free-hand, pairs it with a picture of where the two players actually
are on the board, and prints the exact **`4x4` `{U, D, L, R}` Q matrix**
underneath every case -- always the full grid, never reduced, per the
professor's own correction that the matrix should stay `4x4` and that the
matrix itself, not an entropy number, is the actual output.

![Eight cases, board and Q-matrix graph side by side, from a pure state
through the meeting's own indifference example to an asymmetric mix where
only one player is actually guessing.](figures/png/positions.png)

## Reading the graph

Each node is one cell of the full `4x4` Q matrix (`row = the carrier's
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

## Case 3 — `(1, 1, 1, 0, 1)`: the meeting's own example

> "when we move right and when we move left, there's an equal chance of me
> winning. That's why I am indifferent between the two."

Gap `0.0445`. The carrier's entire live option set is **exactly `{L, R}`**
-- no vertical option survives at all, so this is the plain case the meeting
asked for: two actions that are genuinely equally good against the
defender's own mix (`D 18% / L 82%`). This is the *rarer* shape: only 4 of
94 mixed states cross a purely horizontal pair instead of a vertical one.
Even though only two of its four rows are ever played, the printed matrix is
still the full `4x4` -- `U` and `D` are there to show exactly *why* they lose
out, not omitted.

```
        U        D        L        R
  U   0.139   -0.173    0.099    0.110
  D   0.171    0.185    0.157    0.171
  L   0.810    0.500    0.141    0.810
  R   0.211    0.181    0.211    0.167
```

## Case 4 — `(0, 0, 2, 0, 0)`: a genuine 3-action mix

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

## Case 5 — `(1, 1, 2, 0, 1)`: a near-pure hedge, where rounding would lie

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

## Case 6 — `(5, 1, 6, 1, 1)`: the typical mix, mirrored

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

## Case 7 — `(2, 3, 3, 3, 1)`, on a 5×4 board: this project's own tackle rule

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

## Case 8 — `(0, 2, 1, 2, 0)`: the asymmetric mix

Support `(2, 1)` -- gap `0.0095`. This is the odd one out: the carrier's
equilibrium still mixes two actions (`U 2.6% / D 97.4%`), but the
**defender's** equilibrium is a *single fixed move* (`R`, 100%). Every other
case on this page is a symmetric duel -- both players genuinely guessing.
Here only one side is.

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
split (`2.6% / 97.4%`), not a probability forced by the game the way Case 3's
`18% / 82%` is. This is the meeting's LP-edge-case question made concrete:
**a tie between two rows can look identical, in the printed policy, to a
genuine forced mix, but it is a different phenomenon** -- no best-reply cycle
is involved, and *either* pure `U` or pure `D` alone would also be a valid
equilibrium reply to the defender's fixed `R`. `soccer_nash/certificate.py`'s
`gap` field still distinguishes the two: a tie is a **degenerate** saddle
(row `L`'s pure security value, `0.086`, sits within `0.0095` of the mixed
value), while Case 3's cycle has no pure security value anywhere near it.

## The mathematics behind all eight

A mixed equilibrium is exactly the strategy pair where every action in a
player's support earns the **same expected payoff** against the opponent's
own mix -- that equality is what "indifferent" means in Case 3, and it is
also why the best-response graph above cycles with no resting point: if one
action in the support paid strictly more, that player would raise its weight
on it, which is precisely the condition an equilibrium rules out. A
best-reply cycle and mutual indifference are the same fact seen from two
sides, not two different explanations ([numerics.md](numerics.md) §0). Case 8
shows the boundary of that story: indifference alone (a tie against a fixed
opponent) can produce the same *symptom* -- a fractional policy -- without a
cycle anywhere in the matrix.

## Reproduction

`python scripts/positions.py` prints the exact `4x4` Q matrix and policy for
all eight states and writes `figures/gallery/positions.svg`. A PDF write-up
of this page is at [positions.pdf](positions.pdf).
