# Three mixed states, worked by hand

Requested at the research meeting: stop summarizing the 94 mixed states with a
heatmap or an entropy number, and instead **look at actual states** -- draw
both players on the board with their action probabilities as arrow thickness,
print the *unrounded* stage matrix next to each, and check by eye that mixing
is genuinely required. `scripts/showcase.py` (`make showcase`) does this for
three hand-picked states out of the 94 (7×5 board, 3-cell goal, random move
order, γ = 0.9).

![Three boards: a near-even split at the goal mouth, a far-from-goal 3-way
mix, and a near-pure hedge, each with both players' action probabilities drawn
as weighted arrows.](figures/png/showcase.png)

## Does mixing depend on *where* the players are, or their *relative* position?

The meeting's question, directly: every one of the 94 mixed states has the two
players within **2 cells of each other** (Manhattan distance) -- 40 at distance
1, 54 at distance 2, **zero at distance ≥ 3**. Proximity to the *opponent*, not
distance to the goal, is what forces a guess. The second worked example below
is the sharpest evidence: the carrier is 6 cells from goal (as far as this
board allows) and still faces a genuine 3-way mix, because the defender is
right there.

## State 1 — `(0, 1, 1, 0, 1)`: a near-even split at the goal mouth

Carrier one cell from goal, defender two cells away and able to intercept.
Support `(2, 2)`, entropy **0.96 bits** — close to a fair coin. The carrier
mixes `L 39% / R 61%`; the defender mixes `D 68% / R 32%`. This is the
"genuine 2-action indifference" case the meeting asked to see, as opposed to a
best-reply cycle: rounding to 0.1 would flip the verdict, but the exact gap
(`0.033`) is real and the certificate confirms no pure saddle exists.

```
        U         D         L         R
  U  -0.367    -0.272    -0.190    -0.222
  D  -0.367    -0.272    -0.032    -0.222
  L  -0.302    -0.157    -0.141    -0.211
  R  +0.115    -0.185    -0.190    -0.151
```

## State 2 — `(0, 0, 2, 0, 0)`: far from goal, still a 3-way mix

Carrier at distance 6 from goal (the maximum on this board) with the defender
two cells away and able to intercept. Support `(3, 3)`, entropy **1.11 bits** —
the *richest* mix of the three examples, further from goal than any of them.
The carrier mixes `U 43% / L 55% / R 2%`. This directly answers "would this
happen regardless of place": yes -- what matters is the defender's position
*relative to the carrier*, not the carrier's distance to goal.

```
        U         D         L         R
  U  +0.084    +0.095    +0.103    +0.110
  D  +0.086    +0.076    +0.076    +0.086
  L  +0.086    +0.076    +0.076    +0.086
  R  +0.088    +0.095    -0.088    +0.110
```

## State 3 — `(1, 1, 2, 0, 1)`: a near-pure hedge, where rounding would lie

Support `(2, 2)` again, but entropy only **0.17 bits** — the carrier plays
`R 97.5%`, `D 2.5%`. This is the state to show anyone who asks "is this really
mixed, or just numerical noise": the pure-bound gap is `0.0043`, twelve times
smaller than a 0.1 rounding grid would resolve, yet it is exact and certified
(`soccer_nash/certificate.py`) -- an independent, LP-free check confirms
`maximin < minimax` by exactly that margin. Reading `0.975` as `1.0` would be
the rounding mistake the meeting flagged: real soccer stage games get this
shallow, and treating "almost pure" as "pure" is exactly the trap
[numerics.md](numerics.md) §2 measures at 60-100% of the mixed region.

```
        U         D         L         R
  U  -0.317    -0.211    -0.171    -0.182
  D  -0.317    -0.211    -0.045    -0.182
  L  -0.247    -0.211    -0.157    -0.182
  R  +0.178    -0.167    -0.171    -0.135
```

## How these were picked

`scripts/showcase.py` solves the board once, then filters the 94 no-pure-saddle
states by support size and mixing entropy (`soccer_nash/numerics.mixing_entropy`)
to find a near-even 2-action split, the richest 3-action mix, and the shallowest
genuine gap -- the three qualitatively different ways a stage game ends up
mixed in this game. All three, and every other one of the 94, have a
[machine-checked certificate](result.md) that an independent script re-verifies
without trusting the LP.
