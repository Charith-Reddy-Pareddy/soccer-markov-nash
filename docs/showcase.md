# Six mixed states, worked by hand

Requested at the research meeting: stop summarizing the 94 mixed states with a
heatmap or an entropy number, and instead **look at actual states** -- draw
both players on the board with their action probabilities as arrow thickness,
print the *unrounded* `4x4` stage matrix (`U`, `D`, `L`, `R` only -- no
narrative action names) next to each, and check by eye that mixing is
genuinely required. `scripts/showcase.py` (`make showcase`) does this for six
cases, each a different way a soccer stage game ends up needing two players
who both have more than one live move to guess.

![Six boards: a near-even split at the goal mouth, a far-from-goal 3-way mix,
a near-pure hedge, that state's mirror image, a deterministic-transition mix
forced by reward alone, and the tackle rule's own mixed state -- each with
both players' action probabilities drawn as weighted arrows.](figures/png/showcase.png)

## Does mixing depend on *where* the players are, or their *relative* position?

The meeting's question, directly: every one of the 94 mixed states (canonical
`7x5` board, 3-cell goal, random move order, γ = 0.9) has the two players
within **2 cells of each other** (Manhattan distance) -- 40 at distance 1, 54
at distance 2, **zero at distance ≥ 3**. Proximity to the *opponent*, not
distance to the goal, is what forces a guess. State 2 below is the sharpest
evidence: the carrier is 6 cells from goal (as far as this board allows) and
still faces a genuine 3-way mix, because the defender is right there.

## State 1 — `(0, 1, 1, 0, 1)`: a near-even split at the goal mouth

Carrier one cell from goal, defender two cells away and able to intercept.
Support `(2, 2)`, entropy **0.962 bits** — close to a fair coin. The carrier
mixes `L 39% / R 61%`; the defender mixes `D 68% / R 32%`. This is the
"genuine 2-action indifference" case the meeting asked to see, as opposed to a
best-reply cycle: rounding to 0.1 would flip the verdict, but the exact gap
(`0.033`) is real and the certificate confirms no pure saddle exists.

```
        U         D         L         R
  U  -0.366835 -0.271536 -0.189766 -0.222362
  D  -0.366835 -0.271536 -0.032340 -0.222362
  L  -0.301706 -0.156678 -0.141010 -0.211001
  R   0.114867 -0.185032 -0.189766 -0.150878
```

## State 2 — `(0, 0, 2, 0, 0)`: far from goal, still a 3-way mix

Carrier at distance 6 from goal (the maximum on this board) with the defender
two cells away and able to intercept. Support `(3, 3)`, entropy **1.109
bits** — the *richest* mix of the six examples, further from goal than any of
them. The carrier mixes `U 43.3% / L 54.7% / R 1.9%`. This directly answers
"would this happen regardless of place": yes -- what matters is the
defender's position *relative to the carrier*, not the carrier's distance to
goal.

```
        U         D         L         R
  U   0.083738  0.095109  0.103381  0.110220
  D   0.085599  0.076363  0.076330  0.085599
  L   0.085599  0.076363  0.076330  0.085599
  R   0.088459  0.095109 -0.088213  0.110220
```

## State 3 — `(1, 1, 2, 0, 1)`: a near-pure hedge, where rounding would lie

Support `(2, 2)` again, but entropy only **0.169 bits** — the carrier plays
`R 97.5%`, `D 2.5%`. This is the state to show anyone who asks "is this really
mixed, or just numerical noise": the gap is `0.004261`, twelve times smaller
than a 0.1 rounding grid would resolve, yet it is exact and certified
(`soccer_nash/certificate.py`) -- an independent, LP-free check confirms
`maximin < minimax` by exactly that margin. Reading `0.975` as `1.0` would be
the rounding mistake the meeting flagged: real soccer stage games get this
shallow, and treating "almost pure" as "pure" is exactly the trap
[numerics.md](numerics.md) §2 measures at 60-100% of the mixed region.

```
        U         D         L         R
  U  -0.316945 -0.211001 -0.170790 -0.182213
  D  -0.316945 -0.211001 -0.044962 -0.182213
  L  -0.247069 -0.211001 -0.156678 -0.182213
  R   0.178022 -0.166529 -0.170790 -0.135158
```

## State 4 — `(5, 0, 6, 1, 0)`: the mirror of state 1

`soccer_nash/symmetry.mirror_state` flips the board left-right and swaps which
player carries; this is state 1's exact image under that map. Support
`(2, 2)`, entropy `0.905` bits (state 1's `0.962` computed from the other
player's marginal -- same mix, mirrored). The values negate to machine
precision:

    V(state 1)  = -0.174087
    V(mirror)   = +0.174087
    sum         = +5.55e-17   (zero, to floating-point noise)

This is the cleanest evidence in the whole gallery that the symmetry is
exact, not approximate -- no certificate needed, the residual is 17 orders of
magnitude below the values themselves.

```
        U         D         L         R
  U   0.366835  0.366835 -0.114867  0.301706
  D   0.271536  0.271536  0.185032  0.156678
  L   0.222362  0.222362  0.150878  0.211001
  R   0.189766  0.032340  0.189766  0.141010
```

## State 5 — `(4, 4, 5, 4, 0)`: deterministic transitions, mixing forced by reward alone

Every other example on this page owes its mix to Littman's random move order
-- a transition-level coin. This one has **no transition stochasticity at
all** (`move_order="deterministic"`): the mix comes entirely from
`scoring="territory"` ([territory reward](result.md)), a dense per-step
reward that couples both players' payoffs to their joint action even when the
transition itself is fixed. Support `(2, 2)`, entropy **0.996 bits** — the
single mix closest to a fair coin on this page — with a deep, discount-proof
gap of `0.065321` (0.1-rounding would probably survive here, unlike every
other example above). The carrier mixes `D 53.8% / R 46.2%`; the defender
plays `U 95.1% / L 4.9%`.

```
        U         D         L         R
  U   0.116139  0.104525  0.169846  0.135972
  D   0.094073  0.415483  0.814500  0.450000
  L   0.094073  0.104525  0.000000  0.104525
  R   0.169846  0.500000 -0.670721  0.500000
```

## State 6 — `(2, 3, 3, 3, 1)`: the tackle rule's own mixed state

Solved on a smaller `5x4` board under [the tackle rule](tackle.md)
(`move_order="tackle", tackle_prob=0.5`) instead of Littman's random order:
adjacent players contest the ball with a coin that depends on both players'
actions -- "dive in" or "contain" -- a different mechanism from a random move
order entirely, yet it produces the same 2x2 matching-pennies shape. Support
`(2, 2)`, entropy `0.967` bits, gap `0.022322`. The carrier mixes
`D 60.7% / R 39.3%`; the defender mixes `U 32.1% / D 67.9%`.

```
        U         D         L         R
  U   0.011944 -0.072350  0.018582  0.007062
  D  -0.005242  0.022044  0.080389  0.025575
  L  -0.015879 -0.008641 -0.018582  0.006356
  R   0.041869 -0.000279  0.041869  0.039427
```

## How these were picked

`scripts/showcase.py` solves the canonical board once and filters the 94
no-pure-saddle states by support size and mixing entropy
(`soccer_nash/numerics.mixing_entropy`) to find states 1-3: a near-even
2-action split, the richest 3-action mix, and the shallowest genuine gap --
three qualitatively different ways a stage game ends up mixed under one
resolution rule. States 4-6 extend the set along three more axes: state 4 is
an exact symmetry check (`soccer_nash/symmetry.mirror_state`) rather than a
new mechanism; state 5 swaps the mixing *cause* from transition stochasticity
to reward coupling under fully deterministic transitions; state 6 swaps the
*rule* generating the coin from Littman's random move order to this project's
own tackle rule. All six, and every other mixed state in the underlying runs,
carry a [machine-checked certificate](result.md) that an independent script
re-verifies without trusting the LP.
