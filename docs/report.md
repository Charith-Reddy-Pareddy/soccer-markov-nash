# Pure-First Nash Q-Iteration for the Soccer Markov Game

A research report. Scope and caveats are in `docs/assumptions.md`; every table
below regenerates from `experiments/*.csv` via `scripts/experiments.py`.

---

## 1. Problem

Nash Q-iteration (`Q = R + beta * Nash(Q')`) solves a zero-sum Markov game by
solving a matrix game at every state on every sweep. The matrix-game solve is
the expensive step, and for the CS 540 A10 soccer game it is not obvious it is
even needed: does the game actually require *mixed* strategies anywhere, or does
a pure saddle point exist at every state? The original soccer game (Littman
1994) was introduced precisely because it *does* need mixing "in the place where
they had to have it" -- but where is that place, and does the A10 formulation
inherit it?

## 2. Method

- **Environment** (`soccer_nash/game.py`): the A10 two-player soccer game on a
  configurable grid; state `(x0, y0, x1, y1, b)`; 2380 non-terminal states on
  the 7x5 board. `A10SoccerGame` is the exact-semantics class; `SoccerGame`
  additionally offers two research variants of the collision rule (`random`
  move order, `coinflip` tie-break).
- **Stage solvers** (`soccer_nash/matrix_games.py`, `support_enum.py`):
  - pure-saddle detection (`maximin == minimax`), O(A^2), no LP;
  - LP minimax value (`scipy` HiGHS);
  - support enumeration -- *all* equilibria of a general-sum 2-player game.
- **The pure-first hybrid backup** (`soccer_nash/nash_q.py`): check for a pure
  saddle first; take its value if it exists (it then equals the Nash value);
  fall back to the LP only where it does not.
- **Diagnostics** (`soccer_nash/numerics.py`): the value bracket
  `[min_j (pM)_j, max_i (Mq)_i]` that certifies the LP error; a scale-aware
  `pure` / `degenerate` / `mixed` classifier.
- **Geometry** (`soccer_nash/geometry.py`, `tree.py`): carrier-frame spatial
  features + a small decision tree, to test whether the classification is
  predictable from the state.

## 3. Research questions

- **RQ1 -- Existence.** Under what transition structures does a pure-strategy
  equilibrium exist at every state?
- **RQ2 -- Efficiency.** How much matrix-game work does checking for a pure
  saddle first eliminate?
- **RQ3 -- Mechanism.** Which spatial configurations create states that
  genuinely require mixing?
- **RQ4 -- Numerical robustness.** How do discounting, scale, and classification
  tolerance affect the pure/mixed split?

## 4. Results

### RQ1 -- Existence

| move order | converged stage games with no pure saddle | stationary pure eq? |
|---|---|---|
| `deterministic` (exact A10) | 0 / 2380 | yes (Claim B) |
| `coinflip` tie-break | 0 / 2380 | yes (Claim B) |
| `random` move order | 94 / 2380 (this config) | no |

For the implemented deterministic model every converged stage game has a pure
saddle (Claim A); playing a pure saddle action everywhere is then a stationary
pure-strategy equilibrium (Claim B). A **coin-flip tie-break** -- randomising
who wins a contested square -- does not change this: its value function is
bit-identical to the deterministic game's (`max |V_det - V_coin| = 0` at every
gamma), because optimal play never enters a losing contest, so the coin is never
flipped. Only Littman's **random move order**, where a ball-steal depends on who
resolves first *and* on both players' targets, breaks it.

### RQ2 -- Efficiency

```
random move order, gamma = 0.9 (the case where the LP is actually needed)
          sweeps   matrix-game solves   wall clock   V(kickoff)
pure        18            0                <1 s        +0.000   (a lower bound, not Nash)
hybrid      108        ~9 700              ~14 s        +0.150   (exact)
mixed       108        ~3.6 M              ~8 min       +0.150   (exact, wasteful)
```

The pure-first hybrid solves an LP only on the ~4% of states that lack a pure
saddle -- a ~25x reduction in matrix-game work -- and matches the all-LP value
function to `4e-16`. On the deterministic and coin-flip games it never calls the
LP at all. Two further reductions:

- **Mirror symmetry** (`run_symmetric`, `soccer_nash/symmetry.py`): the game is
  anti-symmetric under board-flip + player-swap, and every state has a distinct
  mirror image, so the 2380 states are 1190 mirror pairs. Solving one per pair
  and reconstructing the other by `V(mirror(s)) = -V(s)` halves the iteration.
- **Freeze-then-iterate** (`run_policy_iteration`): 5x fewer LP solves, but the
  frozen strategies stay maximally stale for 16 outer rounds before locking in,
  so wall-clock is not lower on a game this small. See section 7.

### RQ3 -- Mechanism

**A stage game needs mixed strategies only when the goal mouth is more than one
cell wide** (`experiments/board_sweep.csv`, `goal_mouth_sweep.csv`):

| goal mouth | mixed states (7x5-shaped board) |
|---|---|
| 1 cell | **0** -- every height-3 board, any width 3..11 |
| 2 cells | 186 (5x9 board) |
| 3 cells | 162 |
| >= 4 cells | slowly declines |

With a one-cell goal the carrier's only winning approach is that cell, so the
defender always knows where to stand. With a wider goal the carrier threatens
more than one cell and, under the random move order, the defender cannot cover
them all -- it has to *guess*.

Geometry predicts the classification (`scripts/geometry_model.py`): a depth-4
decision tree separates `mixed` from the rest with **precision 0.96, recall
0.91** using carrier-frame features. The dominant signal is
`defender_can_intercept` -- the defender within one move of the carrier's
forward cell (`P(mixed) = 0.44` vs `0.002`). Of the 94 mixed states, ~88 have
the defender ahead of the carrier and **68 have an equilibrium supported on a
2x2 subgame** with no pure saddle: carrier {advance, hold} against defender
{block, intercept}. Strict dominance collapses only 4 of them fully, so this
describes the equilibrium support, not a reduction of the 4x4 game. All 94 share
one value across every equilibrium (zero-sum interchangeability); 64 have a
unique equilibrium.

### RQ4 -- Numerical robustness

**The value is three numbers** (`value_bracket`): what the row player
guarantees, gets, and can reach. With `scipy` HiGHS they agree to `1.1e-16` per
stage game and `8.7e-10` accumulated -- the LP solution is a certified
`1e-16`-equilibrium.

**Discounting mildly changes which states mix** (`experiments/gamma_sweep.csv`):
the 7x5 random game's mixed count runs 122 -> 94 -> 102 as gamma goes
0.5 -> 0.9 -> 0.995 (non-monotonic); the deterministic game stays 0 at every
gamma. Iterations scale from 26 to 373; the kickoff value rises monotonically
from 0.001 to 0.32.

**The classification is tolerance-robust** (`experiments/tolerance_sweep.csv`):
identical 604 / 1682 / 94 split for every `rel_tol` from `1e-12` to `1e-3`,
degrading only as the tolerance approaches the actual `minimax - maximin` gaps
(80 mixed at `1e-2`, 0 at `1e-1`). A fixed 0.1-rounding scheme -- proposed to
force mixed-strategy indifference -- flips the pure-saddle status of **62**
states, exactly the small-entry mixed region; the deterministic game is safe to
round only because its values are clean `gamma^k` bands well above 0.1.

## 5. Mechanistic explanation

Littman's random move order turns every contested square into a lottery whose
outcome depends on *both* players' concurrent action choices. When the goal is a
single cell, the carrier has one meaningful threat and the defender one
meaningful response, so best responses stay pure. When the goal spans >= 2 cells
and the defender is one interception-move away, the carrier can commit to either
of two threatened cells and the defender must commit to covering one -- a 2x2
matching-pennies subgame. The coin-flip tie-break does not do this because it
randomises the outcome *independently* of the action choices; the deterministic
rule does not because the carrier always wins, removing the guess.

An argument sketch for "one-cell goal => pure everywhere" -- the value collapses
to a one-dimensional race to the single goal cell, in which each player has a
best-response direction independent of the other -- is in `docs/mechanism.md`,
along with 19/19 single-cell configurations that show zero mixed states. It is
not yet a proof.

## 6. Limitations

- The A10 page redacts the ID-specific start position and goal rows; this repo
  uses the standard Littman geometry (configurable). Qualitative results are
  geometry-robust (tested across 40+ board configurations); the exact `3.9%`
  is for the 7x5 / 3-cell-goal case only.
- Several collision sub-cases (carrier vs. stationary opponent, non-carrier
  bump = steal, swap possession) are *interpreted* from the two stated A10
  rules, not quoted. If course staff intended a different rule, Claim A must be
  re-measured; the solver and analysis are unaffected. See `docs/assumptions.md`.
- Claim C (the theoretical A10 game necessarily has a pure equilibrium, over all
  reward/discount settings) is **not** established -- the results are empirical
  enumeration, not a proof.
- The `random` and `coinflip` variants are *different game definitions*, not the
  A10 game; comparisons across them isolate the collision rule and the tie-break
  respectively.

## 7. The algorithm

**Pure-first hybrid Nash Q-iteration.** At each state, highlight best responses
along rows and columns (`O(A^2)`); if one entry is both a row and a column best
response, its value is the Nash value -- use it. Only when no such entry exists
solve the LP. Add mirror-symmetry reduction (solve 1190 of 2380 states) and,
where the equilibrium solve dominates (larger action spaces, general-sum),
freeze-then-iterate. On the A10 game this is exact everywhere and never invokes
an LP; on the random variant it invokes one on 4% of states and matches the
all-LP solution to machine precision.

Freeze-then-iterate results (random game, hybrid solver, same fixed point):

```
                  rounds/sweeps   matrix-game solves   wall clock
value iteration       108             9 672              13.8 s
policy iteration       21             1 879              17.2 s
```

The frozen strategies' *staleness* (distance to the current Nash) stays maximal
for 16 outer rounds, then collapses to `< 1e-8`.

## 8. The equilibrium policy plays correctly (secondary validation)

Self-play and exact best response (`soccer_nash/exploit.py`,
`scripts/selfplay.py`) confirm the solved policy is not merely numerically
satisfying:

- Its duality gap `V0_br(s0) + V1_br(s0)` is `< 1e-9` for every move order -- no
  opponent beats the game value.
- Nash vs. Nash reproduces the value: forced draws in the deterministic and
  coin-flip games; a `+0.157` empirical discounted return (vs. `+0.150`
  computed) in the random game.
- A best response to one assumed opponent is fragile: the Part 2
  best-response-to-the-scripted-opponent policy has exploitability `0.43` --
  worse than moving uniformly at random -- which is exactly the A10 competition's
  warning about non-equilibrium submissions.

## 9. Beyond zero-sum, and open questions

- **General-sum, done.** `soccer_nash/markov_game.py` solves 2-player
  general-sum Markov games by enumerating *all* stage equilibria
  (`support_enum.py`) and selecting one -- the notes' "largest sum of values"
  rule, a no-op for zero-sum but decisive for Battle of the Sexes (it avoids the
  value-destroying mixed equilibrium). It keeps the pure-first philosophy: check
  for a pure Nash before enumerating.
- **The dog / debate game, done.** The research group's second game
  (`soccer_nash/dog_game.py`) has a closed-form Nash -- both players in their
  corners, dog at `w_a*house_a + w_b*house_b` -- and the general-sum solver hits
  it exactly in 1D and 2D, symmetric and asymmetric weights. A finding: *every*
  stage game of the discretised dog game has a pure Nash equilibrium (support
  enumeration is never triggered), so the pure-first strategy pays off here too;
  many states have several pure equilibria, so the selection rule matters.
- **Function approximation, partial.** `soccer_nash/nash_dqn.py` fits a network
  to the stage-game matrices. Even with the A10-format constraints relaxed it
  trails the exact solver on value error and action agreement (section 10); the
  exact solver should stay the ground truth while the continuous-action work
  develops.
- **A proof (open).** Can the "one-cell goal => pure everywhere" observation be
  turned into a theorem about the transition structure?
- **Continuous actions (open).** The dog game's real form has a continuous
  direction/position action space; projected gradient or best-response dynamics
  on the neural Q is the next step, with the discrete solver as the reference.

## 10. Neural Nash-Q vs. the exact solver

`scripts/nash_dqn.py` (7x5 deterministic game, gamma 0.9, 5 -> 96 -> 96 -> 16
regressor with biases, 500 epochs, frozen target network):

| | exact hybrid Nash-Q | neural Nash-Q |
|---|---|---|
| wall clock | 9 sweeps, 0.45 s | 500 epochs, 55 s |
| max `|V - V_exact|` | 0 | 0.54 |
| action agreement | 100% | 53% |
| exploitability | 0 | 0.43 |

The exact solver is both ~120x faster and correct. The network fits the
~1000 zero-value states easily but misses the `gamma^k` structure and the
contested regions -- its policy is about as exploitable as playing uniformly at
random. This is the concrete baseline the eventual continuous-action work has to
beat, and it argues for keeping the exact solver as ground truth rather than
replacing it.
