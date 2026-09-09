# How mixed are the mixed states, and what is mixing worth?

`scripts/mixing.py` (`make mixing`), on the 7×5 board with a 3-cell goal, random
move order, γ = 0.9. `experiments/mixing.csv` has the per-state numbers.

## 1. The mixes are shallow

All 94 no-pure-saddle stage games have a `minimax − maximin` gap **under 0.07**
(median 0.029). These sit just outside the pure-saddle region -- a defender who
plays their maximin action loses only a few percent of value by not mixing. None
is a wild anti-coordination game; the deepest is `0.070`.

Because the stage games are this close to pure saddles, the *exact* equilibrium
mixing weights are numerically delicate -- a `1e-10` change in the converged
value function swings a reported `p*` -- so this page does not quote per-state
weights. The structural facts below are robust (they use `>` comparisons, not
the LP vertex).

## 2. Existence is not the same as amount

The carrier's **mixing entropy** (`mixing_entropy`, `numerics.py`): 0 for a pure
strategy, 1 bit for an even 2-way / matching-pennies mix, more for a genuinely
spread strategy.

- **median 0.55 bits** over the 94 states -- most are near-pure hedges
  (≈ 87/13), where the carrier mostly commits and randomises a little.
- **26 of 94** reach `≥ 0.9` bits -- a near-even split. These are the states
  with genuine indifference, the ones with the largest `minimax − maximin`
  gaps.
- max 1.27 bits -- one 3-action mix.

So the mixed region is 94 states, but the **strongly-mixed core is ~26**. The
"value of mixing" (§3) and the entropy agree: the shallow states are shallow
because the carrier barely has to randomise there.

## 3. The support is matching pennies

| support | states | structure |
|---|---|---|
| 2 × 2 | 68 | **all 68 are matching pennies** -- the carrier's best reply crosses between the two defender columns and vice versa |
| 3-action (one side) | 26 | near-pure saddles (2×1 effective support) or one 3×3 |

This is the same 68 / 94 the geometric templates cover
([templates.md](templates.md)); the carrier randomizes between two scoring lanes
(climb / advance) and the defender between covering them.

## 4. The value of mixing

`V(hybrid) − V(pure maximin)` -- how much a player gives up by being unable to
mix:

- **at the 94 mixed states:** mean `+0.126`, median `+0.107`, up to `+0.367`.
- **propagated to the kickoff** `(0, 2, 6, 2, 0)`: `+0.150` with mixing vs
  `0.000` restricted to pure strategies. A pure-strategy player at this kickoff
  can only *secure a draw*; the mixing option is worth a decisive `+0.15` edge.

So even though each individual mixed stage game is shallow, they compound: the
carrier meets a mixed-required state on most paths to the goal, and a defender
who never mixes leaks value at each one. That is the practical case for solving
the LP on the 3.95% of states that need it rather than falling back to the
maximin bound everywhere.
