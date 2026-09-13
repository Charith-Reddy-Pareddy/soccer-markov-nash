# Not every fractional LP output is a forced mixture

"94 no-pure-saddle states" is a precise, certified count: the value-bracket
gap (`certify_game`) is nonzero at exactly 94 states on the canonical board.
It is *not* the same claim as "94 states where mixing is forced with a
uniquely determined split" -- [positions.md](positions.md) found four cases
(4, 5, 9, 10) where an action outside the LP's reported support ties its
value exactly, meaning the reported split is one point on a larger,
equally-valid set of equilibria, not a number the game forces. This page
answers the natural follow-up: how many of the 94, exactly?

`scripts/degeneracy.py` checks every no-pure-saddle state, both players, for
a zero-weight action that ties the reported support's value (`tol=1e-6`, the
same tolerance `support_shape` and `positions.py`'s `_support` already use),
and sorts each state into one of three categories:

| category | what it means | count |
|---|---|---|
| **unique** | neither side has a tied zero-weight action -- the reported support is the whole indifference class, forced by the matching-pennies structure with no freedom left over | 64 (68.1%) |
| **degenerate equilibrium face** | a side with >= 2 reported actions also has a zero-weight action tied with it -- the LP found one vertex of a larger polytope face | 16 (17.0%): 8 carrier, 8 defender |
| **pure reply tied inside the equilibrium set** | a side reported as a *single* action (nominally "pure") has a zero-weight action tied with it -- that "pure" reply isn't uniquely forced either | 14 (14.9%), always the defender, never the carrier |

64 + 16 + 14 = 94; no state is degenerate on both sides at once.

(This is a different "degenerate" than [numerics.md](numerics.md)'s
`classify_stage_game` taxonomy, which reserves "degenerate saddle" for
*pure*-saddle states with more than one tied optimal row/column. Both are the
same underlying idea -- more than one equally-valid solution -- applied to
different halves of the pure/no-pure-saddle split.)

## The three questions this separates

1. **Does a state have no pure saddle?** -- the certificate answers this with
   no LP at all (`maximin < minimax`). This is what "94" counts.
2. **Does the LP output have fractional support?** -- yes, for every
   genuinely mixed side, by construction (that's what "mixed" means for that
   side). Not interesting on its own.
3. **Is that fractional support strategically necessary** -- forced by
   keeping the opponent indifferent, with no other split possible -- or
   merely **degenerate**, a tie the LP happened to resolve one particular
   way? This page answers that one directly, per state, and the answer is
   "necessary" 68.1% of the time and "degenerate" the other 31.9%.

## Reading the two degenerate kinds

**Degenerate equilibrium face** ([positions.md](positions.md) Cases 4, 5):
the *reported* support already has two or more actions, and it turns out a
further, unweighted action ties them exactly. Case 5's carrier is the
sharpest example: the LP reports `U 43.3% / L 54.7% / R 1.9%`, but `D` ties
all three exactly too -- the true indifference class is all four actions,
and the 43.3/0/54.7/1.9 split printed is one point on a whole face of
equally-good carrier mixes, not a uniquely forced ratio.

**Pure reply tied inside the equilibrium set** ([positions.md](positions.md)
Case 9): the *reported* support is a single action -- the defender's
equilibrium is printed as pure `R`, 100% -- but `D` ties `R`'s value exactly.
The defender's choice of `R` over `D` is exactly as arbitrary as the
carrier's own `U`/`D` tie in the same state; it just isn't printed as a
percentage because the LP put all the weight on one side. **This pattern is
always on the defender's side, never the carrier's**, across all 14 states
that show it -- consistent with the separate finding that the defender's
support is never larger than the carrier's ([positions.md](positions.md)):
a defender that is already close to indifferent between its narrower set of
options is more likely to have an unweighted tie sitting just outside it.

## What doesn't change

The 2x2 matching-pennies core ([result.md](result.md) §3) is a property of
the *matrix* -- two rows and two columns whose relative payoffs cross both
ways -- and is present in all 94 states regardless of degeneracy; that is
what makes a pure saddle impossible in the first place, and it is the
certificate's job to prove, not the LP's job to report. Degeneracy is a
separate, LP-vertex-specific fact about *which* equilibrium got printed. Both
things can be true of the same state at once, and for 30 of the 94 (Cases 4,
5, 9, 10 among them), they are.

## Reproduction

`python scripts/degeneracy.py` prints the breakdown and writes
`experiments/degeneracy.csv` (one row per no-pure-saddle state: support
sizes, which side is degenerate, and the category).
