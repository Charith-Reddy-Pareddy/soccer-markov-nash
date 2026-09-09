# Reproducing Littman (1994) Table 3

Littman's central experiment: train four soccer policies -- **MR / MM** by
minimax-Q, **QR / QQ** by ordinary Q-learning -- and score each against a random
opponent, a hand-built opponent, and a **challenger** trained specifically to
beat it. His result (Table 3):

| | vs. random | vs. hand-built | vs. its challenger |
|---|---|---|---|
| MR (minimax) | 99.3% won | 48.1% | 35.0% |
| MM (minimax) | 99.3% | 53.7% | 37.5% |
| QR (Q-learning) | 99.4% | 26.1% | **0.0%** |
| QQ (Q-learning) | 99.5% | 76.3% | **0.0%** |

> "Every deterministic offense in this game has a perfect defense, much like
> rock, paper, scissors."

`scripts/tournament.py` (`make tournament`) reproduces the *structure* with an
**exact solver** -- no learning, no Monte-Carlo noise -- on Littman's board
(5×4, two-cell goals, five actions including `stand`, random move order, goal
reset, γ = 0.9). Four player-0 policies:

- **minimax** -- the exact Nash / minimax-Q fixed point (`nash_q` hybrid)
- **greedy/rand** -- best response to a uniform-random opponent (Littman's QR)
- **greedy/self** -- the deterministic maximin-security policy (a QQ-style
  deterministic fixed point)
- **hand-built** -- Littman's "simple rules for scoring and blocking" policy
  (§6.2): on offense drive at the goal; on defense stand on the square the
  carrier wants next (`soccer_nash/opponents.py` `handbuilt_policy`). It beats a
  random opponent ~76% of games -- a competent deterministic bot.

scored by player 0's exact **expected discounted goal difference** from the
kickoff.

## Result

| policy | vs. random | vs. hand-built | vs. challenger | robustness gap |
|---|---|---|---|---|
| **minimax** | +0.67 | +0.28 | **+0.15** | +0.52 |
| greedy/rand | **+0.92** | +0.68 | **−0.59** | +1.51 |
| greedy/self | +0.05 | 0.00 | −0.00 | +0.05 |
| hand-built | +0.72 | 0.00 | **−0.68** | +1.39 |

(the `scoring="win"` view -- `P(win) − P(loss)` -- tells the same story: minimax
`+0.57 → +0.17` against its challenger, greedy/rand `+0.59 → −0.29`, hand-built
`+0.51 → −0.42`.)

Three regimes, exactly as in the paper:

1. **minimax exploits *and* survives.** It beats the random and hand-built
   opponents by a wide margin *and* stays ahead (`+0.15` goal difference)
   against the worst possible opponent -- because the Nash policy is
   unexploitable (duality gap 0). Its robustness gap is the smallest of any
   policy that also wins against weak opponents.
2. **greedy/rand exploits but dies.** It is the strongest policy against a
   random or scripted opponent -- `+0.92`, better than minimax -- but a
   challenger drives it from dominant to *losing* (`−0.59`). This is Littman's
   "QR won 0.0%": a deterministic offense has a pure counter.
3. **greedy/self is safe but toothless; hand-built is neither.** The
   maximin-security policy is unexploitable (`−0.00` vs. its challenger) but
   never presses an advantage (`+0.05` vs. a random opponent). Littman's
   hand-built policy beats a random opponent well (`+0.72`) yet a challenger
   turns it into a `−0.68` loss -- deterministic, so it has a pure counter, just
   like the greedy policies.

Figure: `docs/figures/gallery/tournament.svg`.

## Why minimax is the only policy in the top-left

The mixed states are matching-pennies stage games ([templates.md](templates.md)).
A greedy policy commits to one row of each; the challenger then plays the column
that beats that row, every time -- the rock-paper-scissors trap Littman names.
The minimax policy randomizes over the support with exactly the weights that
make the challenger indifferent, so there is no column to punish it with. This
is the practical payoff of solving the LP on the [41% of the equilibrium
path](occupancy.md) that needs it, rather than falling back to a deterministic
best reply.
