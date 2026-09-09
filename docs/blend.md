# Sweeping the move-resolution rule

`docs/design.md` treats the three move-resolution rules as discrete design
choices. `move_order="blend"` makes it continuous: each joint action is resolved
by **Littman's random order with probability `blend`**, and by the
**deterministic carrier-wins rule** otherwise. `blend = 0` is the deterministic
game, `blend = 1` is the random game, and `blend` sweeps between them.

`scripts/blend.py` (`make blend`) solves the game exactly along the sweep.

## A sharp threshold at `blend = 0.5`, with an overshoot

7×5 board, 3-cell goal, `γ = 0.9`:

| `blend` | 0.00–0.50 | 0.52 | 0.56 | 0.60 | 0.70 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|
| no-pure-saddle states | **0** | 36 | 122 | 120 | 140 | 92 | 94 |

5×4 board, 2-cell goal:

| `blend` | 0.00–0.55 | 0.58 | 0.65 | 0.70 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|
| no-pure-saddle states | **0** | 28 | 80 | **88** | 68 | 48 | 56 |

Two things stand out, on both boards:

1. **Mixing needs the random component to be the majority.** For `blend ≤ 0.5`
   every stage game still has a pure saddle -- the deterministic carrier-wins
   tie-break, even at 50% weight, is enough to give the carrier a
   weakly-dominant response and collapse the matching-pennies structure. The
   switch is sharp: 0 mixed at `blend = 0.50`, dozens at `blend = 0.52`.

2. **The count overshoots just past the threshold.** Around `blend ≈ 0.6–0.7`
   there are ~1.5× *more* no-pure-saddle stage games than in the fully random
   game. Near the boundary many stage games sit right on the pure/mixed edge, so
   more of them tip over; as `blend → 1` the value function settles and the
   count relaxes to its 94 (7×5) / 56 (5×4).

This sharpens [discussion.md](discussion.md) §3: it is not just that Littman's
random *order* (rather than the tested action-independent coin) creates mixing --
here the random order also has to **outweigh** the deterministic component of
the resolution rule. A minority of random-order resolution is washed out.

**What this is and is not.** `blend` interpolates *two particular* transition
mechanisms, `P = (1−p)·P_det + p·P_random`. The statement is: *in this
interpolation family*, mixing first appears sharply above `p ≈ 0.5`. That is a
property of this family, **not** a theorem and **not** a universal threshold for
mixing arbitrary transition rules. Whether `p_c = 0.5` has a structural
explanation -- and whether the overshoot is a genuine phase transition with a
critical exponent -- is open ([discussion.md](discussion.md) §7).

The deterministic-limit value `V(kickoff) = 0` (a forced draw) holds for all
`blend ≤ 0.5` and rises smoothly once mixing turns on.
