# When Does a Soccer Markov Game Need Mixed Strategies?

*A pure-first Nash-Q approach: efficiency, numerical stability, and geometric
structure.*

A research report. Scope and caveats are in `docs/assumptions.md`; every table
below regenerates from `experiments/*.csv` (`make experiments phase dqn`).

Two contributions:

1. **Algorithmic** -- a pure-first hybrid Nash-Q backup: check each stage game
   for a pure saddle, take its value when it exists, fall back to the LP only
   where it does not. Exact, and it skips the LP on 96&ndash;100% of states.
2. **Structural** -- a characterization of *when* a stage game is intrinsically
   mixed: it takes a stochastic resolution order, a goal mouth wider than one
   cell, and interception geometry. The 94 mixed states of the 7x5 random game
   reduce to a handful of matching-pennies templates ([templates.md](templates.md)).

The two meet in RQ2: the hybrid is fast *because* the structural result says
almost every stage game is pure, so the LP is rare.

![Kickoff: player 0 (blue) carries the ball toward the right goal; player 1
(green) defends the left.](figures/kickoff.svg)

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

## 2. The soccer Markov game and Nash-Q

- **Environment** (`soccer_nash/game.py`): the A10 two-player soccer game on a
  configurable grid; state `(x0, y0, x1, y1, b)`; 2380 non-terminal states on
  the 7x5 board. `A10SoccerGame` is the exact-semantics class; `SoccerGame`
  additionally offers two research variants of the collision rule (`random`
  move order, `coinflip` tie-break).
- **Nash-Q** (`soccer_nash/nash_q.py`): the operator `Q = R + gamma * Nash(Q')`,
  where `Nash` is the minimax value of the stage game `M_s`. Value iteration
  applies it every sweep; freeze-then-iterate (policy iteration) freezes the
  stage strategies, runs cheap linear evaluation sweeps, then re-solves.
- **Stage solvers** (`soccer_nash/matrix_games.py`, `support_enum.py`):
  pure-saddle detection (`maximin == minimax`, O(A^2), no LP); the LP minimax
  value (`scipy` HiGHS); support enumeration -- *all* equilibria of a
  general-sum 2-player game.
- **Diagnostics** (`soccer_nash/numerics.py`): the value bracket
  `[min_j (pM)_j, max_i (Mq)_i]` that certifies the LP error; a five-way
  `pure` / `degenerate` / `mixed` classifier whose only load-bearing class is
  "genuine mixed" (&sect;5, RQ4).
- **Geometry** (`soccer_nash/geometry.py`, `tree.py`, `templates.py`):
  carrier-frame spatial features, a decision tree, and a canonical-state
  template analyzer, to ask whether the classification is predictable and
  reducible.

## 3. The pure-first hybrid algorithm

At each state, mark the best responses along rows and columns (`O(A^2)`); if one
entry is both a row and a column best response, its value is the Nash value --
use it. Only when no such entry exists solve the LP. On the deterministic A10
game a pure saddle exists at every state, so the LP is never called; on the
random variant it is called on 3.95% of states, and the value function matches
the all-LP solution to `4e-16`.

Two further reductions:

- **Mirror symmetry** (`run_symmetric`, `soccer_nash/symmetry.py`): the game is
  anti-symmetric under board-flip + player-swap, and every state has a distinct
  mirror image, so the 2380 states are 1190 mirror pairs. Solving one per pair
  and reconstructing the other by `V(mirror(s)) = -V(s)` cuts the deterministic
  solve from 0.33 s to 0.23 s (median of 5; deterministic dynamics only -- the
  mirror of a stochastic transition is not verified).
- **Freeze-then-iterate** (`run_policy_iteration`): on the random game, 108
  value-iteration sweeps and 9 672 LP solves become 21 policy-iteration rounds
  and 1 879 LP solves -- 5x fewer -- but the frozen strategies stay maximally
  stale for 16 rounds before locking in, so wall-clock is not lower on a game
  this small (17.2 s vs 13.8 s). It pays off only where the equilibrium solve
  dominates (larger action spaces, general-sum).

## 4. Research questions

- **RQ1 -- Existence.** Under what transition structures does a pure-strategy
  equilibrium exist at every state?
- **RQ2 -- Efficiency.** How much matrix-game work does checking for a pure
  saddle first eliminate?
- **RQ3 -- Mechanism.** Which spatial configurations create states that
  genuinely require mixing?
- **RQ4 -- Numerical robustness.** How do discounting, scale, and classification
  tolerance affect the pure/mixed split?

## 5. Results

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

Two numbers, kept separate. **LP-call rate** is a property of the game and
reproduces exactly; **wall-clock** is a median over 5 repeats and depends on the
machine and the LP backend (`scripts/benchmark.py`).

```
random move order, gamma = 0.9 (the case where the LP is actually needed)
                    sweeps   LP calls   states needing LP   LP/state/sweep   median wall clock
pure                  18          0           --                 --               0.3 s   (a lower bound, not Nash)
hybrid               108       9 672        3.95% (94/2380)      0.038            13.2 s  (exact)
hybrid + mirror       -           -            -                  -                 -     (deterministic game only)
all-LP (mixed)       108     ~3.6 M         100%                 1.0             ~8 min  (exact, wasteful)
```

The pure-first hybrid solves an LP only on the **3.95%** of states that lack a
pure saddle -- a **~25x** per-sweep reduction in LP calls -- and matches the
all-LP value function to `4e-16`. On the deterministic and coin-flip games it
calls the LP zero times.

*Do not conflate the LP-call reduction with the wall-clock speedup.* The 3.95%
figure is exact and portable. The wall-clock ratio (~13 s vs ~8 min here, ~37x)
also reflects how much of each sweep is matrix construction rather than the
solve, the LP backend, and the machine; treat it as indicative, not a headline.
The mirror and freeze-then-iterate reductions are in &sect;3.

### RQ3 -- Mechanism

**Goal-mouth width, not board size, is the gate.** The phase diagram
(`scripts/phase_diagram.py`, `experiments/phase_diagram.csv`) sweeps every board
with `3 <= width <= 11`, `3 <= height <= 9`, `width * height <= 45`, against goal
widths 1 to `height - 2`:

![Mixed-state fraction by board size and goal-mouth width. The goal-width-1
column is zero for every board.](figures/phase_diagram.svg)

- **Goal width 1: 0 mixed states -- on all 40 one-cell configs.**
- **Goal width >= 2: mixed states -- on all 87 wider-goal configs**, at a
  fraction of 2.7-12% (mean 5%).

`scripts/phase_diagram.py --analyze` decomposes the variance. Whether a board
has *any* mixed state is a **perfect** function of goal width (1 vs >= 2). Among
the boards that do, an OLS of the mixed *fraction* is carried by board area
(R^2 0.64 -- a dilution effect, more midfield filler), with goal width adding
little (R^2 0.27 alone) and a *negative* fitted coefficient. So goal width
creates or removes mixing; it does not scale it, and board size only dilutes it.
Every state is reachable from the kickoff (`reachable == states` in every row),
so the fraction over reachable states equals the fraction over all states.

Geometry predicts the classification (`scripts/geometry_model.py`): a depth-4
decision tree separates `mixed` from the rest with **precision 0.96, recall
0.91** using carrier-frame features. The dominant signal is
`defender_can_intercept` -- the defender within one move of the carrier's
forward cell (`P(mixed) = 0.44` vs `0.002`).

Beyond prediction, the 94 mixed states canonicalize under the board mirror to 47
pairs and cluster into **8 geometric templates** (`scripts/templates.py`,
`docs/templates.md`). The **four templates with a 2x2 equilibrium support are
all verified matching pennies** -- the row player's best reply flips between the
two defender columns and vice versa -- covering **68 of 94** states; carrier
{climb, advance} against defender {cover a lane, hold the forward cell}. The
other 26 are borderline near-pure saddles (22) or one 3x3 mix (4). All 94 share
one value across every equilibrium (zero-sum interchangeability); 64 have a
unique equilibrium.

The precise analytic condition for an unavoidable mixed stage game is a
stochastic (½–½) resolution order **and** a goal mouth wider than one cell
**and** a defender close enough to contest the carrier's forward cell but unable
to cover every scoring lane in a single move. Remove any one -- a one-cell goal,
deterministic resolution, the coin-flip tie-break -- and every stage game has a
pure saddle.

### RQ4 -- Numerical robustness

**A stage game falls into one of five classes** (`classify_stage_game`), and the
count that matters -- "genuine mixed" -- is the one robust to how the line is
drawn:

| class | test | on the 7x5 random game |
|---|---|---|
| exact pure saddle | `maximin == minimax` exactly | rare (clean `gamma^k` cells) |
| numerically-indistinguishable saddle | `minimax - maximin <= eps * scale` | most of the 2286 non-mixed |
| strict pure saddle | + a unique maximin row and minimax column | 604 |
| degenerate saddle | + several tied rows or columns (e.g. all-zeros) | 1682 |
| genuine mixed | `minimax - maximin > eps * scale` | **94** |

The tolerance `eps` scales with the matrix entries, so a stage game deep in the
discounted bands is judged the same way as one at the goal. "Genuine mixed"
is the only class that forces an LP.

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

## 6. Mechanistic explanation

Littman's random move order turns every contested square into a lottery whose
outcome depends on *both* players' concurrent action choices. When the goal is a
single cell, the carrier has one meaningful threat and the defender one
meaningful response, so best responses stay pure. When the goal spans >= 2 cells
and the defender is one interception-move away, the carrier can commit to either
of two threatened cells and the defender must commit to covering one -- a 2x2
matching-pennies subgame. The coin-flip tie-break does not do this because it
randomises the outcome *independently* of the action choices; the deterministic
rule does not because the carrier always wins, removing the guess.

**One goal cell (`docs/proof.md`).** For a single goal cell per side, the
defender has a closed-form optimal strategy -- reach the goal row, then slide
along it toward the one goal cell, never onto the carrier -- verified to secure
`V*` from every state, which proves `minimax(M_s) = V*(s)` by weak duality. And
every single-cell stage game is solvable by iterated weak-dominance elimination
(`soccer_nash/dominance.py`), hence has a pure saddle -- machine-checked for
every board up to 11x5 and every discount 0.5-0.99, 0 exceptions. This proves
the theorem for each finite board; a board-size-free argument for the carrier's
half is still open.

## 7. Secondary validation: the equilibrium policy plays correctly

Self-play and exact best response (`soccer_nash/exploit.py`,
`scripts/selfplay.py`) confirm the solved policy is not merely numerically
satisfying:

- Its duality gap `V0_br(s0) + V1_br(s0)` is `< 1e-9` for every move order -- no
  opponent beats the game value.
- Nash vs. Nash reproduces the value: forced draws in the deterministic and
  coin-flip games; a `+0.162 +/- 0.002` empirical discounted return over 5 seeds
  of 3000 games (vs. `+0.164` computed) in the random game.
- A best response to one assumed opponent is fragile: the Part 2
  best-response-to-the-scripted-opponent policy has exploitability `0.43` --
  worse than moving uniformly at random -- which is exactly the A10 competition's
  warning about non-equilibrium submissions.
- The A10 deliverables (Part 1 successor/reward tables, Part 2 imitation network,
  competition networks) are in `docs/a10_*.md`. Over 5 seeds
  (`experiments/a10_competition_seeds.csv`): the Part 2 imitation network plays
  an optimal action at `0.990 +/- 0.000` of states and wins the Q8 rollout every
  time; the competition Network Second is exact (`1.000` optimal, `0`
  exploitable) but Network First sits at `0.994` optimal and a
  `0.19 +/- 0.02` exploitability from this kickoff -- one badly-fit state a
  best-responder can exploit, and it does not wash out with re-seeding.

## 8. Neural Nash-Q vs. the exact solver

The meeting's stated pipeline is: exact discrete solver first, then a network to
replicate it. `scripts/nash_dqn.py` fits a `5 -> 96 -> 96 -> 16` regressor with
biases (frozen target network, 600 epochs) to the exact stage matrices of the
7x5 deterministic game and measures the gap over **5 seeds**
(`experiments/nash_dqn_seeds.csv`):

| metric | exact hybrid Nash-Q | neural Nash-Q (mean +/- sd, n=5) |
|---|---|---|
| value / policy | ground truth | -- |
| max `\|V - V_exact\|` | 0 | 0.55 +/- 0.02 |
| mean `\|V - V_exact\|` | 0 | 0.13 +/- 0.00 |
| action agreement | 100% | 43% +/- 2% |
| pure/mixed classification agreement | 100% | 67% +/- 2% |
| exploitability (duality gap) | `<1e-9` | 0.44 +/- 0.05 |
| convergence (epochs to MSE plateau, of 600) | exact fixed point | 469 +/- 31; still creeping down at 600 in 3/5 seeds |
| runtime | 9 sweeps, 0.3 s | 600 epochs, ~35 s |

Deterministic game only -- `train_nash_dqn` rejects the stochastic variants, so
the network is never asked to represent a genuinely mixed stage game. Even so it
trails: the network fits the ~1000 zero-value states easily (small *mean* error)
but misses the `gamma^k` bands and the contested regions -- worst-state value
error `0.55`, policy about as exploitable as moving uniformly at random, and it
mis-calls *whether* a state needs mixing a third of the time. This is the
concrete baseline the eventual continuous-action work has to beat, and it is why
the exact solver stays the ground truth rather than being replaced.

## 9. Beyond zero-sum, and open questions

- **General-sum, done.** `soccer_nash/markov_game.py` solves 2-player
  general-sum Markov games by enumerating *all* stage equilibria
  (`support_enum.py`) and selecting one -- the notes' "largest sum of values"
  rule, a no-op for zero-sum but decisive for Battle of the Sexes, where it
  avoids the mixed equilibrium whose value is below either pure one. It keeps
  the pure-first philosophy: check for a pure Nash before enumerating. The
  soccer game itself is zero-sum, so this is an extension point rather than a
  change to the main result.
- **Function approximation, partial.** `soccer_nash/nash_dqn.py` fits a network
  to the stage-game matrices (&sect;8). It trails the exact solver and only
  handles the deterministic game; the exact solver should stay the ground truth
  while any continuous-action work develops.
- **A proof (partly done).** The single-cell pure-saddle theorem now has the
  defender's optimal strategy in closed form and a dominance-solvability
  certificate for every finite board (`docs/proof.md`); a board-size-free proof
  of the carrier's half is open.

## 10. Limitations

- The A10 page redacts the ID-specific start position and goal rows; this repo
  uses the standard Littman geometry (configurable). Qualitative results are
  geometry-robust: across the phase-diagram sweep, every one-cell-goal board has
  0 mixed states and every wider-goal board has some. The exact `3.95%` mixed
  fraction is for the 7x5 / 3-cell-goal case only.
- Several collision sub-cases (carrier vs. stationary opponent, non-carrier
  bump = steal, swap possession) are *interpreted* from the two stated A10
  rules, not quoted. If course staff intended a different rule, Claim A must be
  re-measured; the solver and analysis are unaffected. See `docs/assumptions.md`.
- Claim C (the theoretical A10 game necessarily has a pure equilibrium, over all
  reward/discount settings) is **not** established for the general goal mouth --
  those results are enumeration. For the *single-cell* goal it is partly proved:
  the defender's half in closed form, the whole per board by dominance
  elimination (`docs/proof.md`).
- The `random` and `coinflip` variants are *different game definitions*, not the
  A10 game; comparisons across them isolate the collision rule and the tie-break
  respectively.

**Reproducibility.** The value-iteration results are exact dynamic programming
and carry no seed. The randomised measurements are each run over 5 seeds and
quoted as mean +/- sd: the neural Nash-Q metrics (`nash_dqn_seeds.csv`), the
A10 imitation and competition networks (`a10_part2` / `a10_competition --seeds`),
the random-game self-play return (`+0.162 +/- 0.002`), and the wall-clock
figures (median of 5 repeats). LP-call counts and mixed-state counts are
deterministic.
