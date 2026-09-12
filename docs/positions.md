# Player positions and the 4×4 stage game, drawn the way the meeting sketched them

At the research meeting the professor drew a stage game by hand: a small grid
of cells, each marked, with arrows chasing each other around the grid to show
that no cell is a stable outcome. `scripts/positions.py` (`make positions`)
turns that sketch into a real diagram, generated from the exact solver instead
of drawn free-hand, pairs it with a picture of where the two players actually
are on the board, and prints the exact `4x4` `{U, D, L, R}` matrix underneath
-- the professor's own correction that the matrix, not an entropy number, is
the actual output.

![Seven cases, board and stage-game graph side by side, from a pure state
through the meeting's own indifference example to this project's own tackle
rule.](figures/png/positions.png)

## Reading the graph

Each node is one surviving cell of the stage game (`row = the carrier's
action`, `col = the defender's action`, rows and columns always reoriented so
the carrier's payoff is what's printed, regardless of which player id
actually carries -- see `_oriented_matrix` in `scripts/positions.py`). A blue
arrow points from a cell to the cell in the same column with the higher
payoff -- the carrier's own reason to switch rows. A green arrow points from a
cell to the cell in the same row with the *lower* payoff -- the defender's
reason to switch columns, since the defender minimises. Only cells that
survive [iterated dominance](../soccer_nash/numerics.py) are drawn
(`essential_subgame`), so a `4x4` game collapses to whatever's actually
load-bearing.

- **A pure state has exactly one node with no outgoing arrow.** That cell is
  the saddle; everything else eventually points to it.
- **A mixed state has arrows everywhere.** Every node points somewhere, so
  the arrows chase each other around the grid with no resting point.

## Case 1 — `(4, 0, 5, 0, 0)`: a pure state, for contrast

Support `(1, 1)` -- a strict saddle at `U/U`. The carrier plays `U` always;
the defender answers `U` always; neither has any reason to deviate. Nothing
in this matrix requires a solver more sophisticated than "check every cell
once."

```
        U        D        L        R
  U   0.234    0.317    0.523    0.272
  D   0.151    0.211    0.211    0.222
  L   0.161    0.182    0.172    0.190
  R   0.012   -0.095    0.058    0.098
```

## Case 2 — `(0, 1, 1, 1, 0)`: the typical mix

Support `(2, 2)`, gap `0.0136`. The carrier mixes `U 63.5% / D 36.5%`; the
defender answers `U 36.5% / R 63.5%`. **This is the common shape**: the
carrier is deciding which goal row to head for, and the defender is guessing
which one. 90 of the 94 mixed states on the canonical board cross a vertical
pair exactly like this one.

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

Support `(2, 2)`, gap `0.0445`. The carrier's entire live option set is
**exactly `{L, R}`** -- no vertical option survives at all, so this is the
plain case the meeting asked for: two actions that are genuinely equally good
against the defender's own mix (`D 18% / L 82%`). This is the *rarer* shape:
only 4 of 94 mixed states cross a purely horizontal pair instead of a
vertical one.

```
        U        D        L        R
  U   0.139   -0.173    0.099    0.110
  D   0.171    0.185    0.157    0.171
  L   0.810    0.500    0.141    0.810
  R   0.211    0.181    0.211    0.167
```

## Case 4 — `(0, 0, 2, 0, 0)`: a genuine 3-action mix

Support `(3, 3)`, gap `0.0047`, entropy `1.109` bits -- the richest mix on
this page. Six cells from goal, as far as this board allows, with the
defender close enough to threaten three different rows at once: no single row
is safely better than the other two, so three actions are simultaneously
undominated. Requested explicitly at the meeting: the matrix itself, not
entropy, is the point.

```
        U        D        L        R
  U   0.084    0.095    0.103    0.110
  D   0.086    0.076    0.076    0.086
  L   0.086    0.076    0.076    0.086
  R   0.088    0.095   -0.088    0.110
```

## Case 5 — `(1, 1, 2, 0, 1)`: a near-pure hedge, where rounding would lie

Support `(2, 2)` again, but entropy only `0.169` bits -- the carrier plays
`R 97.5%`, `D 2.5%`. Reading `0.975` as `1.0` would be the rounding mistake
the meeting flagged: the gap (`0.0043`) is real and certified, twelve times
smaller than a 0.1 rounding grid would resolve.

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

This is direct evidence that Case 2's shape is not a one-off coincidence of
where it happens to sit on the board -- the identical mix reappears, exactly
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
mechanism with nothing to do with move order at all. It reduces to the same
2×2 duel shape (support `(2, 2)`, gap `0.0223`): the carrier mixes
`D 60.7% / R 39.3%`, the defender mixes `U 32.1% / D 67.9%`. Different cause,
identical structure.

```
        U        D        L        R
  U  -0.012    0.005    0.016   -0.042
  D   0.072   -0.022    0.009    0.000
  L  -0.019   -0.080    0.019   -0.042
  R  -0.007   -0.026   -0.006   -0.039
```

## The mathematics behind all seven

A mixed equilibrium is exactly the strategy pair where every action in a
player's support earns the **same expected payoff** against the opponent's
own mix -- that equality is what "indifferent" means in Case 3, and it is
also why the best-response graph above cycles with no resting point: if one
action in the support paid strictly more, that player would raise its weight
on it, which is precisely the condition an equilibrium rules out. A
best-reply cycle and mutual indifference are the same fact seen from two
sides, not two different explanations ([numerics.md](numerics.md) §0).

## Reproduction

`python scripts/positions.py` prints the exact stage matrix and policy for
all seven states and writes `figures/gallery/positions.svg`. A PDF write-up
of this page is at [positions.pdf](positions.pdf).
