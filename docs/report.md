# When Does a Soccer Markov Game Need Mixed Strategies?

*A pure-first Nash-Q approach: efficiency, numerical stability, and geometric
structure.*

A research report on a configurable family of two-player soccer Markov games
([design.md](design.md)). Scope and caveats are in `docs/assumptions.md`; how
every number was produced is in [methods.md](methods.md); the open questions and
what the mixed region depends on are in [discussion.md](discussion.md). Every
table regenerates from `experiments/*.csv`.

## Position relative to Littman (1994)

Littman **already established** that the soccer Markov game can require mixed
strategies: his Figure 2 gives a concrete state where any deterministic
offensive choice can be blocked indefinitely, and only randomising between
*stand* and *move* guarantees an opening. This project does **not** rediscover
that. It asks the next question:

> **Where does mixing appear, why does it appear, and what parameter switches it
> on?**

Littman demonstrated that mixed strategies are necessary in a *particular*
soccer Markov game (5×4 grid, five actions incl. `stand`, random move order,
discount/draw interpretation). We investigate what structural features of the
soccer environment determine *when* such mixed equilibria arise, and whether
their computational cost can be avoided except where mixing is genuinely
necessary. `docs/littman.md` reproduces his Figure 2 and his Table 3.

## Two contributions

1. **Algorithmic.** A pure-first hybrid Nash-Q backup: at each stage game check
   for a pure saddle (`O(A²)`, no LP), use its value when one exists, fall back
   to the LP only where it does not. Exact everywhere. Its portable claim is
   that **it eliminates LP work on almost every stage game** (0% of states on
   the deterministic game, 3.95% on the random one); the observed wall-clock
   speedup is reported separately because it also depends on the machine and the
   LP backend.
2. **Empirical structural characterization.** Mixed equilibria in the studied
   soccer family are not distributed arbitrarily: across the tested boards they
   are localized to states combining a *stochastic move order*, a *multi-cell
   goal mouth*, and a *defender with insufficient one-step coverage*, and most
   such states reduce to matching-pennies-like 2×2 supports. The region is
   invariant to the kickoff and to how the horizon is scored (`win` vs `rate`),
   but not to a dense per-step reward that couples to both actions
   ([discussion.md](discussion.md) §5); it carries 41% of the equilibrium-path
   occupancy despite being 4% of the state space ([occupancy.md](occupancy.md)).
   This is an empirical characterization over the tested parameter grid, not a
   theorem.

## Central hypothesis

Mixed equilibria in the studied soccer family are not distributed arbitrarily
across the state space; they arise from a specific interaction between goal-mouth
geometry and stochastic move-order coupling. Because most states retain pure
saddles, a pure-first hybrid Nash-Q solver should obtain the exact minimax value
while substantially reducing LP usage.

## Objective definitions

The report compares several objectives; they are related but not
interchangeable, so:

| where | objective |
|---|---|
| A10 Part 1 / exact | **undiscounted, finite 100-step horizon** — `run_finite_horizon`, non-stationary `V` |
| stationary Nash-Q | **discounted infinite-horizon fixed point**, `γ < 1` (`run()`); a contraction proxy for the above |
| `scoring="rate"` variant | **continuing discounted** objective — a goal resets, value = expected discounted goal difference ([reward.md](reward.md)) |
| `scoring="territory"` variant | `win` **plus a dense per-step reward** for the ball in the opponent's final third — the reward-side analogue of a stochastic transition ([reward.md](reward.md)) |
| Littman replication | Littman's **discount / draw** interpretation, `γ = 0.9` ≡ a 0.1 per-step draw probability |

Unless a section says otherwise, results are the stationary discounted fixed
point at `γ = 0.9`, and RQ4 shows the pure/mixed classification is stable across
`γ ∈ [0.5, 0.995]` and matches the exact undiscounted answer.

![Kickoff: player 0 (blue) carries the ball toward the right goal; player 1
(green) defends the left.](figures/png/kickoff.png)

---

## 1. Problem

Nash Q-iteration (`Q = R + beta * Nash(Q')`) solves a zero-sum Markov game by
solving a matrix game at every state on every sweep. The matrix-game solve --
a linear program -- is the expensive step. Littman's soccer game needs mixed
strategies *somewhere*; the questions are **where**, **why**, and whether the LP
can be skipped everywhere else. A discrete minimax-Q solver on the usual A10
formulation, in fact, returns a *pure* strategy at every state -- the reason
turns out to be the collision rule, not the solver (§6, `docs/numerics.md`).

## 2. The soccer Markov game and Nash-Q

- **Environment** (`soccer_nash/game.py`): the A10 two-player soccer game on a
  configurable grid; state `(x0, y0, x1, y1, b)`; 2380 non-terminal states on
  the 7x5 board. `A10SoccerGame` is the exact-semantics class; `SoccerGame`
  additionally offers two research variants of the collision rule (`random`
  move order, `coinflip` tie-break).
- **Nash-Q** (`soccer_nash/nash_q.py`): the operator `Q = R + gamma * Nash(Q')`,
  where `Nash` is the minimax value of the stage game `M_s`. The A10 game itself
  is *undiscounted* (`+1` / `-1` at a goal, `0` otherwise, tie after 100 steps),
  so its exact solution is 100-step backward induction (`run_finite_horizon`,
  RQ1). `run()` with `gamma < 1` is a faster stationary approximation -- a
  contraction device -- and RQ4 shows the pure/mixed split holds across `gamma`
  in `0.5-0.995`; `gamma = 0.9` unless noted. Freeze-then-iterate (policy
  iteration) freezes the stage strategies, runs cheap linear evaluation sweeps,
  then re-solves.
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

Littman's minimax-Q solves an LP at every `(s, a, o)` update. This solver adds
five speed-ups, none in the paper:

- **Precompute the outcome table.** Build every stage game's transition list
  once (`NashQIteration.__init__`), then look up `s'` during the recursion --
  the professor's "solve once for every `s`".
- **Pure-first.** At each state, mark the best responses along rows and columns
  (`O(A^2)`); if one entry is both a row and a column best response, its value
  *is* the Nash value -- use it, no LP. On the deterministic game a pure saddle
  exists at every state (LP never called); on the random variant the LP runs on
  3.95% of states, and the value function matches the all-LP solution to `4e-16`
  -- a **~25x** reduction in LP calls.
- **LP-result memoization.** Key the LP value on `round(M, 11).tobytes()`; near
  convergence the same stage matrices repeat sweep-to-sweep, trimming a further
  few percent (`~10 000 -> ~9 700` calls).
- **Back up `p M q`** in freeze-then-iterate (below), not the bounds -- 2x fewer
  outer rounds.

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

![Line chart: value iteration's Bellman residual climbs to 9 digits over about
100 sweeps.](figures/png/convergence.png)

**Which value quantity to back up in the frozen sweep** (the professor's
specific question). The frozen `(p, q)` don't match the current `Q`, so does the
evaluation sweep use `p^T M q`, `min_j (pM)_j`, or `max_i (Mq)_i`? All three
reach the same fixed point, but `p^T M q` converges in **half the rounds** (21
vs 41-42) -- it is the unbiased estimate while the strategies are stale, where
the two bounds are systematically pessimistic / optimistic and stretch the
thrash phase from 16 rounds to 28 (`run_policy_iteration(eval_value=...)`,
[numerics.md](numerics.md) §3).

![Staleness per outer round for the three backup choices; p-M-q locks in near
round 16, the two bounds near round 28.](figures/png/eval_value.png)

Value iteration's Bellman residual decays geometrically at rate ≈ γ.
Freeze-then-iterate makes no progress until the frozen stage saddle flips --
concurrent stochastic games have no monotone policy improvement
([discussion.md](discussion.md) §4).

## 4. Research questions

- **RQ1.** When does the soccer Markov game admit pure versus mixed stage
  equilibria?
- **RQ2.** Can pure-saddle detection reduce Nash-Q computation without changing
  the exact minimax value?
- **RQ3.** What spatial and transition structures characterize the
  mixed-equilibrium states?
- **RQ4.** How sensitive are equilibrium classification and computation to
  discounting, numerical tolerance, and reward formulation?

## 5. Results

![Summary: goal width branches to PURE-only vs candidate mixed states; with a
stochastic move order and a defender that can't cover both lanes, the best
replies cross into a 2x2 matching-pennies stage game that needs the LP. Beside
it, the pure / hybrid / mixed triangle: deterministic pure = hybrid = mixed,
random order pure &lt; hybrid = mixed; and 3.95% of states carry 41% of the
equilibrium path.](figures/png/story.png)

### RQ1 -- Existence

The A10 game is undiscounted, so its exact solution is 100-step backward
induction (`run_finite_horizon`, `experiments/undiscounted.csv`):

| move order | mixed stage games (state x step, of 238 000) | V(kickoff) | pure equilibrium |
|---|---|---|---|
| `deterministic` (exact A10) | **0** | 0.000 | yes -- pure (non-stationary) |
| `coinflip` tie-break | **0** | 0.000 | yes |
| `random` move order | 3 148 (404 states, some step) | +0.459 | no |

For the deterministic A10 game **every stage game at every horizon has a pure
saddle**, and the equilibrium is *constructive and memoryless*: each player's
win attractor (the states from which it forces a goal) coincides exactly with
the `V* = +/-1` set, and the pure profile "attractor move on your winning set,
safety move elsewhere" (`soccer_nash/attractor.py`, `scripts/positional.py`)
realizes `V*` at every state -- the deterministic game is positionally
determined. The discount was never load-bearing: the stationary `gamma < 1`
solve agrees at every `gamma` from 0.5 to 0.995 (0 mixed stage games), with
these counts:

| move order | converged stage games with no pure saddle, `gamma = 0.9` | stationary pure eq? |
|---|---|---|
| `deterministic` | 0 / 2380 | yes |
| `coinflip` | 0 / 2380 | yes |
| `random` | 94 / 2380 (this config) | no |

A **coin-flip tie-break** -- randomising who wins a contested square -- does not
change this: its value function is bit-identical to the deterministic game's
(`max |V_det - V_coin| = 0`), because optimal play never enters a losing
contest, so the coin is never flipped. Only Littman's **random move order**,
where a ball-steal depends on who resolves first *and* on both players' targets,
breaks it.

*The scope of this claim.* In the particular soccer family studied here, this
one coin-flip collision variant does not create new mixed stage games, whereas
random move ordering does. This is not a statement about all action-independent
stochastic transitions -- stochastic transitions can certainly alter strategic
values and equilibrium structure in other games. The specific mechanism here is
that the random *order* couples the ball-steal outcome to *both* players'
concurrent action choices (§6).

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

**The portable claim: the hybrid eliminates LP work on almost every stage
game** -- 100% of states on the deterministic game, 96.05% on the random one --
while matching the all-LP value function to `4e-16`. That is a property of the
game, not the machine.

*Reported separately, not as a headline:* the observed wall-clock ratio
(~13 s vs ~8 min here) also reflects matrix construction, the LP backend, and
the machine. Treat it as indicative. The mirror and freeze-then-iterate
reductions are in §3.

### RQ3 -- Mechanism

**Across the tested family, a one-cell goal produced no mixed stage games while
every tested goal width ≥ 2 produced some.** The phase diagram
(`scripts/phase_diagram.py`, `experiments/phase_diagram.csv`) sweeps
**127 configurations**: boards with `3 ≤ width ≤ 11`, `3 ≤ height ≤ 9`,
`width·height ≤ 45`, against goal widths `1` to `height − 2`.

![Mixed-state fraction by board size (rows) and goal-mouth width (columns),
127 configurations (3 ≤ W ≤ 11, 3 ≤ H ≤ 9, W·H ≤ 45). The goal-width-1 column is
zero for every board.](figures/png/phase_diagram.png)

- **Goal width 1: 0 mixed stage games — on all 40 tested one-cell configs.**
- **Goal width ≥ 2: at least one mixed stage game — on all 87 tested wider-goal
  configs**, at a fraction of 2.7–12% (mean 5%).

The precise statement: **a multi-cell goal is necessary for the mixed region in
the studied family, and across our tested boards it is sufficient for at least
one mixed state to exist.** It is *not* a proof that every board with a
multi-cell goal must have a mixed state, nor that every state on such a board is
mixed.

`scripts/phase_diagram.py --analyze` decomposes the variance. Over the tested
grid, whether a board has *any* mixed state is a perfect function of goal width
(1 vs ≥ 2). Among the boards that do, an OLS of the mixed *fraction* is carried
by board area (R² 0.64 — a dilution effect, more midfield filler), with goal
width adding little (R² 0.27 alone) and a *negative* fitted coefficient. Every
state is reachable from the kickoff, so the fraction over reachable states
equals the fraction over all states.

Geometry predicts the classification (`scripts/geometry_model.py`): a depth-4
decision tree separates `mixed` from the rest with **precision 0.96, recall
0.91** using carrier-frame features. The dominant signal is
`defender_can_intercept` -- the defender within one move of the carrier's
forward cell (`P(mixed) = 0.44` vs `0.002`). Sharper still (`scripts/showcase.py`,
[showcase.md](showcase.md)): **every one of the 94 mixed states has the two
players within 2 cells of each other** (Manhattan distance -- 40 at distance 1,
54 at distance 2, none at distance ≥ 3). Distance to the *opponent*, not
distance to the goal, is what forces a guess -- one worked example is a
3-action mix 6 cells from goal, the farthest this board allows, because the
defender is adjacent.

Beyond prediction, the 94 mixed states canonicalize under the board mirror to 47
pairs and cluster into **8 geometric templates** (`scripts/templates.py`,
`docs/templates.md`). The **four templates with a 2x2 equilibrium support are
all verified matching pennies** -- the row player's best reply flips between the
two defender columns and vice versa -- covering **68 of 94** states; the
carrier choosing between two of `{U, D, L, R}` against the defender covering
one of the same two (exact letters per template in [templates.md](templates.md)).
The other 26 are borderline near-pure saddles (22) or one 3x3 mix (4). All 94 share
one value across every equilibrium (zero-sum interchangeability); 64 have a
unique equilibrium.

![One board per matching-pennies template: carrier and defender each with two
probability-weighted arrows.](figures/png/templates.png)

**A per-stage-game certificate.** `soccer_nash/certificate.py` re-derives the
pure/mixed label from the stage matrix with `O(A^2)` arithmetic and no LP, and
`scripts/verify_mechanism.py` (`make verify`) runs it over the reachable state
space. For a single goal cell every stage game gets a saddle-cell certificate
(0 mixed, cross-checked against `onecell.certify`); for every wider goal, **each
mixed stage game gets a certificate that an independent checker re-verifies to
`~1e-16`**: the `maximin < minimax` gap (the LP-free proof that no pure saddle
exists), a closed best-reply cycle, the exact equilibrium with
`epsilon_equilibrium < 1e-15`, and a **2x2 matching-pennies submatrix — present
in all 94 of the 7x5 mixed states and all 56 of the 5x4 ones**, a strictly
stronger statement than the 68-of-94 support count above. The switch is checked
invariant across `gamma in {0.5, 0.9, 0.995}`, 4 vs 5 actions, `win` vs `rate`
scoring, and exact undiscounted backward induction (16/16 configs), and a
`+/-1e-6` perturbation of every stage matrix flips no genuine classification
(smallest gap `~4e-3`). [result.md](result.md) is the precise scope-limited
statement, the certificate description, and the prior-work positioning;
`experiments/mechanism_certificate.json` is the machine output.

![A mixed stage game's 2x2 matching-pennies core drawn as a payoff matrix: each
player's best reply flips with the other's choice, so the best replies cycle and
no cell is a pure saddle.](figures/png/mechanism.png)

Every policy and value surface in this report is drawn in `docs/gallery.html`
(`make gallery`, from `soccer_nash/viz.py`): policy fans, the mixing map above,
value heatmaps, and the same state under two resolution rules.

**The candidate mechanism** (empirical, not proved): a stage game requires
mixing when a **stochastic (½–½) resolution order** meets a **multi-cell goal**
(so the carrier has two scoring lanes) and a **defender that can contest the
forward cell but not cover both lanes in one move**. Under those conditions the
carrier's best "which lane" reply flips with the defender's cover and vice
versa — *crossing best replies*, hence no pure saddle:

> geometric condition ⟹ crossing best replies ⟹ no pure saddle

Drop any one clause — a one-cell goal, deterministic resolution, the coin-flip
tie-break — and every tested stage game has a pure saddle. Each tested config is
now machine-certified in both directions ([result.md](result.md),
`make verify`); `docs/mechanism.md` gives the interpretable argument for the
single-cell case. What is *not* settled is a board-size-free theorem for
arbitrary `W, H` (§ open questions).

*How far the goal-width switch generalizes* (`scripts/generalize.py`,
[generalize.md](generalize.md)). It survives board scale (checked to `11×5` and
`9×7`), aspect ratio, and goal placement — non-contiguous goal cells like
`(1, 3)` or `(0, 2, 4)` still produce mixed stage games, so adjacency does not
matter, only that the carrier has two cells the defender cannot cover at once.
It does **not** survive a change of transition family. Adding action-independent
movement noise (`SoccerGame(slip=p)`) produces mixed stage games even under
deterministic resolution and even with a single goal cell, and so does this
project's own `tackle` rule — the defender commits to a challenge that wins the
ball with probability `tackle_prob` or bounces off ([tackle.md](tackle.md)). So
the switch is a property of Littman's move-order rule, where the unresolved coin
only bites where the carrier has two lanes; `slip` and `tackle` supply that coin
everywhere. Reading the `2×2` cores of the `7×5` mixed states, the carrier's
crossing pair is a vertical move in 90 of 94 — it is choosing which goal row to
head for, and the defender is guessing it.

![Six small boards, one per collision rule (deterministic, coinflip, random,
blend, slip, tackle), the defender pinned at the goal mouth and every carrier
cell shaded by how far its stage game is from a pure saddle. Deterministic and
coinflip are blank; random and blend light a thin band by the goal; slip and
tackle light a broad region around the defender.](figures/png/rule_fingerprints.png)

Interpolating the resolution rule (`move_order="blend"`: random order with
probability `blend`, else deterministic) shows the onset is a **sharp
threshold**, not a ramp: 0 no-pure-saddle stage games for `blend ≤ 0.5`, dozens
at `blend = 0.52`, and an overshoot to ~1.5× the fully-random count near
`blend ≈ 0.7` (`scripts/blend.py`, [blend.md](blend.md)). *In this particular
interpolation family* `P = (1−p)·P_det + p·P_random`, mixing first appears
sharply above ≈ 0.5 — the random order has to *outweigh* the deterministic
tie-break. Whether `p_c = 0.5` has a structural explanation is open; it is a
property of this family, not a universal threshold.

The mixed states are also where the game is actually played: under the Nash
policy only 456 of 2380 states are reachable from the kickoff, and the 94
no-pure-saddle states carry **41%** of the discounted occupancy
(`soccer_nash/occupancy.py`, [occupancy.md](occupancy.md)).

#### How mixed, and worth how much

*Existence is not the same as amount.* Every one of the 94 no-pure-saddle stage
games is within `0.07` of a pure saddle (`minimax − maximin`, median `0.029`),
and the carrier's **mixing entropy** has median `0.55` bits — most are near-pure
hedges (≈ 87/13), not real randomisation. Only **26 of 94** reach `≥ 0.9` bits
(near an even 2-way split); those are the states with genuine indifference. So
the mixed region is 94 states, but the *strongly* mixed core is ~26.

The robust structural facts: 68 have a 2×2 matching-pennies support, and the
**value of mixing** — `V(hybrid) − V(pure maximin)` — averages `+0.13` at those
states. It compounds along a path: at the centred kickoff a pure-strategy player
secures only a draw where mixing is worth `+0.15` (`scripts/mixing.py`,
[mixing.md](mixing.md)).

#### Which reward objectives move the mixed region

Three objectives on the same dynamics (`scripts/reward.py`, `make reward`,
[reward.md](reward.md)): `win` (first goal ends it), `rate` (goal scores `±1`
and play continues from a restart, value = expected discounted goal
difference), and `territory` (this project's own -- `win` plus a dense per-step
reward for holding the ball in the opponent's final third). 7×5, 3-cell goal,
random order, `γ = 0.9`:

| objective | no-saddle states | shared with `win` | value range | V(kickoff) |
|---|---|---|---|---|
| `win` | 94 | -- | `[−1.00, +1.00]` | +0.150 |
| `rate` | 94 | **94 (0 in, 0 out)** | `[−0.88, +0.88]` | +0.132 |
| `territory` (0.05) | 112 | 73 (39 in, 21 out) | `[−1.00, +1.00]` | +0.000 |

**`win` and `rate` give the identical region.** A mixed NE is an equilibrium of
one stage game `M(s)`; changing how the *horizon* is scored rescales `M(s)`
through `V(s′)` but does not cross the `maximin = minimax` boundary. The value
*range* compresses under `rate` because conceding restarts play with possession.

**`territory` moves it** -- the per-step reward is added *directly* to
`M(s)[a0, a1]`, and it depends on where the ball ends up, i.e. on *both*
players' actions. That action-coupled term crosses the boundary: 39 states
enter, 21 leave at `territory_reward = 0.05`, and the drift grows with the
reward. Sharper still, `territory` gives the **deterministic** game 69 genuinely
mixed stage games where `win` and `rate` give 0 -- coupling the immediate reward
to both actions does what Littman's move order does by coupling the transition.
`slip` / `tackle` are the transition-side versions of the same effect
([generalize.md](generalize.md), [tackle.md](tackle.md)); `territory` is the
reward-side one.

![Three value heatmaps, win / rate / territory. Win and rate have the same
gradient with the rate extremes pulled toward the kickoff value; territory's
surface is warped upward across the attacking third.](figures/png/reward_value.png)

#### Littman's fifth action

Littman's 1994 soccer game adds a `stand` action, and his Figure 2 (a carrier
pinned against its own goal) is the canonical "soccer needs mixing" example.
Solving Littman's 5×4 board with `n_actions=5` (`scripts/littman.py`,
`make littman`):

| action set | no-saddle states | stand in the mixed support | V(kickoff) |
|---|---|---|---|
| N, S, E, W | 56 | 0 | +0.147 |
| N, S, E, W, stand | 56 | 48 | +0.172 |

The stand action does **not** move where mixing happens (48 of 56 no-pure-saddle
states are shared) -- it changes *what* the mix is. The carrier hedges with
`STAND` rather than a move action, and every player is slightly better for the
option. The Figure 2 state (0,1,1,1,1) is a clean matching-pennies matrix with
equilibrium `(½ U, ½ STAND)` for both players. The deterministic game keeps a
pure saddle everywhere with five actions too. Details: [littman.md](littman.md).

![Littman Figure 2: the same state with four moves (a U/L mix) and five
(a U/STAND mix).](figures/png/littman_stand.png)

### RQ4 -- Numerical robustness

**The soccer game contains rock-paper-scissors.** The pure/mixed check is the
professor's O(A²) best-reply method: mark player 0's best row in each column and
player 1's best column in each row; a cell with both is a pure saddle, and if
none has both the best replies *cycle* and mixing is forced -- no LP needed. The
stage game at `(0, 1, 1, 1, 1)` (carrier pinned against its own goal, defender
adjacent -- Littman's Figure 2 geometry) has exactly the crossing structure of
rock-paper-scissors.

![Rock-paper-scissors and a soccer stage game side by side, best replies marked;
in both the row and column best replies never coincide.](figures/png/rps_vs_soccer.png)

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
guarantees (`min_j (pM)_j`), gets (`p^T M q`), and can reach (`max_i (Mq)_i`) --
the quantities the professor flagged as never numerically identical. With `scipy`
HiGHS they agree to `1.1e-16` per stage game and `8.7e-10` accumulated -- the LP
solution is a certified `1e-16`-equilibrium, so the definitional ambiguity is
not a practical problem. If it were, the row player takes the *guaranteed* value
`min_j (pM)_j` -- always a safe lower bound on the true minimax.

**Discounting mildly changes which states mix** (`experiments/gamma_sweep.csv`):
the 7x5 random game's mixed count runs 122 -> 94 -> 102 as gamma goes
0.5 -> 0.9 -> 0.995 (non-monotonic); the deterministic game stays 0 at every
gamma. Iterations scale from 26 to 373; the kickoff value rises monotonically
from 0.001 to 0.32.

**The classification is tolerance-robust** (`experiments/tolerance_sweep.csv`):
identical 604 / 1682 / 94 split for every `rel_tol` from `1e-12` to `1e-3`,
degrading only as the tolerance approaches the actual `minimax - maximin` gaps
(80 mixed at `1e-2`, 0 at `1e-1`). A fixed 0.1-rounding scheme -- proposed to
force mixed-strategy indifference -- is **never safe**: across every discount
from 0.5 to 0.99 it wrongly collapses 60-100% of the mixed stage games, because
the genuine `minimax - maximin` gaps never reach the 0.1 grid (heavy discounting
shrinks the `gamma^k` differences; light discounting leaves entries barely
apart). The deterministic game is safe to round only because its values are
clean `gamma^k` bands well above 0.1.

![Line chart: fixed 0.1-rounding wrongly makes 60-100% of the mixed stage games
look pure at every discount, and their true gap never reaches
0.1.](figures/png/discounting_trap.png)

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

**One goal cell — supporting theory, not a central claim** (`docs/proof.md`).
For a single goal cell per side the defender has a closed-form strategy — reach
the goal row, slide toward the goal cell, never onto the carrier — machine-
verified to secure `V*` from every state (so `minimax(M_s) = V*(s)` by weak
duality), and every single-cell stage game is iterated-weak-dominance-solvable
(machine-checked, all boards ≤ 11×5, all discounts 0.5–0.99, 0 exceptions). This
supports the mechanism above per finite board. Per the meeting feedback, a
board-size-free proof is *not* being pursued as a deliverable; the attractor /
dominance code stands as conjecture-plus-evidence.

## 7. Secondary validation: the equilibrium policy plays correctly

Self-play and exact best response (`soccer_nash/exploit.py`,
`soccer_nash/evaluate.py`, `scripts/selfplay.py`) confirm the solved policy is
not merely numerically satisfying:

**Littman's Table 3, reproduced exactly** (`scripts/tournament.py`,
[tournament.md](tournament.md)). Four player-0 policies scored on Littman's board
(5×4, 5 actions, random order, goal reset, γ = 0.9), against a random opponent,
Littman's hand-built "scoring and blocking" policy (§6.2, which beats a random
opponent ~76% of games), and a per-policy best-response challenger:

| policy | vs. random | vs. hand-built | vs. challenger | robustness gap |
|---|---|---|---|---|
| minimax (Nash) | +0.67 | +0.28 | **+0.15** | +0.52 |
| greedy vs. random | **+0.92** | +0.68 | **−0.59** | +1.51 |
| greedy self-play | +0.05 | 0.00 | −0.00 | +0.05 |
| hand-built | +0.72 | 0.00 | **−0.68** | +1.39 |

Only the minimax policy both presses an advantage against a weak opponent *and*
stays ahead of its challenger; every deterministic policy -- greedy or
hand-built -- does exactly one. This is Littman's "every deterministic offense
has a perfect defense, like rock-paper-scissors": the mixed states are
matching-pennies stage games, and a deterministic policy hands the challenger a
column to punish.

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

## 8. Exploratory: function approximation and general-sum

*Groundwork for the eventual continuous-action work, not equal-weight
contributions.* The exact solver is the point of both — it is the ground truth.

### Two neural baselines, and where the approximation breaks

On the deterministic 7×5 game, over **5 seeds** (`experiments/nash_dqn_seeds.csv`):
a **Q net** regresses toward the stage matrices `Q(s, a0, a1)` and extracts a
policy by minimax; a **policy net** regresses *directly* onto the exact
equilibrium strategies `(p(s), q(s))`.

| (5 seeds) | exact | Q net (mean ± sd) | policy net |
|---|---|---|---|
| max `\|V − V_exact\|` | 0 | 0.55 ± 0.02 | — |
| action agreement | 100% | 43% ± 2% | **99.9% ± 0.0%** |
| pure/mixed classification | 100% | 67% ± 2% | — |
| max equilibrium regret `ε_NE` | 0 | — | 0.46 ± 0.02 |
| exploitability (duality gap) | `<1e-9` | 0.44 ± 0.05 | **0.84 ± 0.00** |

**The policy net names the exact-optimal action at 99.9% of states and is
*more* exploitable than the mediocre Q net** (duality gap 0.84 vs 0.44). Two
reasons: the ~0.1% wrong states are exactly the ones a best-responder attacks
(the rock-paper-scissors trap again), and a softmax head necessarily smears what
should be pure strategies into exploitable near-indifference. So the failure is not primarily
value approximation — "name the right action" is not "play a Nash". Even on a
small discrete game where the exact solution is a 0.3 s computation, a
straightforward neural approximation does not preserve game-theoretic
robustness. Keep the exact solver as ground truth.

### General-sum — a sanity check, not a second paper

`soccer_nash/markov_game.py` extends pure-first beyond zero-sum: enumerate *all*
stage equilibria (`support_enum.py`), select one by "largest sum of values". On
Battle of the Sexes this correctly avoids the mixed equilibrium whose value is
below either pure one; the soccer game is zero-sum, so it does not touch the
main result. Future work.

## 9. Open questions

Full statements and best answers in [discussion.md](discussion.md) §§1–7. The
ones that matter most:

- **Turn the mechanism into a theorem.** *Geometric condition ⟹ crossing best
  replies ⟹ no pure saddle* is verified over the tested grid; proving the
  forward implication for a general multi-cell goal (probably via
  concurrent-reachability positional determinacy for the single-cell case, then
  a splitting argument) is the most valuable next step.
- **Is the goal-width dichotomy already known** in the stochastic /
  pursuit-evasion literature? This determines how strongly novelty can be
  claimed and is a question for the professor / a literature pass.
- **The `blend` threshold.** Is `p_c ≈ 0.5` structural, and is the overshoot a
  phase transition? ([blend.md](blend.md), [discussion.md](discussion.md) §7)
- **Reward-invariance, precisely.** The mixed region is identical under `win`
  and `rate` (horizon rescaling) but moves under `territory` (a dense per-step
  reward). The conjecture: it is invariant to any change that only rescales
  `M(s)` through `V(s′)` — potential-based shaping, monotone rescaling of `V` —
  and moves under any reward with a term depending on both actions
  ([reward.md](reward.md), [discussion.md](discussion.md) §7).
- **Move order vs. randomness** — answered for this family (§ RQ1, RQ3): the
  tie-break coin does not create mixing here; the move order does.
- **Freeze-then-iterate thrashing** — answered (§ RQ2): concurrent stochastic
  games have no monotone strategy improvement.

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

---

## 11. What weight each experiment carries

Not all of the experiments are equal-weight contributions.

**Core.**
- pure vs. mixed by collision rule (RQ1)
- the pure-first hybrid: LP eliminated on 96–100% of stage games, exact (RQ2)
- the phase diagram: goal width, not board size, gates the mixed region (RQ3)
- the geometry → templates → 2×2 matching-pennies reduction (RQ3)
- numerical robustness: the three value numbers, the rounding trap, the
  classifier (RQ4)

**Strong supporting evidence.**
- occupancy: 4% of states, 41% of the equilibrium path
- Littman Table 3 reproduced: minimax exploits *and* survives; greedy does one
- reward-objective comparison: the mixed region is invariant, the value range
  is not
- symmetry: value *and* policy equivariance where the equilibrium is unique
- the `blend` interpolation and its threshold

**Future / exploratory.**
- the two neural baselines (§8)
- the general-sum extension (§8)
- freeze-then-iterate and the `eval_value` choice
- the single-cell dominance / attractor proof machinery

## 12. Related work — what this project stands on

- **A. Markov games and minimax-Q.** Littman (1994) defines the two-player
  zero-sum Markov game, replaces the MDP `max` with a minimax computed by an LP,
  and introduces soccer *because* its optimal policy can be probabilistic. This
  project takes that as the baseline and asks where and why.
- **B. Concurrent / reachability games.** de Alfaro–Henzinger–Kupferman and
  related work on positional determinacy of concurrent reachability games — the
  frame for the deterministic game's attractor result and the single-cell
  argument.
- **C. Stochastic games and stationary equilibria.** Shapley (1953); Filar &
  Vrieze; the orderfield-property literature (Parthasarathy–Raghavan) — the
  frame for "one target cell vs. many".
- **D. Strategy improvement / policy iteration.** Hoffman–Karp, Condon for
  turn-based games; the *absence* of a monotone improvement guarantee for
  concurrent games explains the freeze-then-iterate thrash.
- **E. Reward shaping in multi-agent RL.** Ng–Harada–Russell potential-based
  shaping — the frame for the `shaping.py` result that PBRS leaves the
  equilibrium exact.

*What is prior work:* mixed strategies can be necessary in soccer (A); pure-first
enumeration before LP (the meeting). *What this project adds:* an empirical map
of where the mixed region is, what parameter switches it on, how much of the
equilibrium path it covers, and a hybrid solver that pays LP cost only there.
[result.md §6](result.md) states the goal-width result against Littman (1994),
the ordered-field property, concurrent reachability games, and pursuit-evasion
in detail.

## 13. Package map

The code groups conceptually as:

```
core        game.py  markov_game.py
equilibrium matrix_games.py  nash_q.py  support_enum.py  numerics.py
analysis    geometry.py  tree.py  templates.py  occupancy.py  reachability.py
            symmetry.py  attractor.py  dominance.py  onecell.py  certificate.py
learning    nash_dqn.py  mlp.py
evaluation  exploit.py  evaluate.py  simulate.py  opponents.py  best_response.py
render      render.py  viz.py
```
