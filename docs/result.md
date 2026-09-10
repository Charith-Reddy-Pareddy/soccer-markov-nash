# The goal-width switch for mixed equilibria

**One-line statement.** In the random-move-order soccer Markov game, the
equilibrium is pure at *every* state when each side defends a single goal cell,
and *provably requires mixing at some state* the moment the goal is two or more
cells wide -- and this switch is invariant to the discount factor, the action
set, and the reward objective.

This page is the precise, scope-limited version of the "RQ3 -- mechanism"
result in [report.md](report.md). Everything here is machine-checked by
`scripts/verify_mechanism.py` (`make verify`, ~90 s), which writes
`experiments/mechanism_certificate.json`.

---

## 1. The claim, precisely

Fix the transition rule to Littman's **random move order**: a joint action
`(a0, a1)` is resolved by ordering the two players uniformly at random and
applying their moves in sequence, where moving into an occupied cell fails and
transfers the ball. Let `V*` be the discounted minimax value function (Shapley
fixed point) and, for each non-terminal state `s`, let

    M_s[a0, a1] = E_order[ r0(s, a0, a1) + gamma * V*(s') ]

be the converged stage game (player 0 maximises, player 1 minimises).

> **Result (verified over the tested family).**
>
> 1. **Single goal cell** (`goal_rows` a singleton). Every reachable `M_s` has a
>    pure-strategy saddle point: `maximin(M_s) = minimax(M_s) = V*(s)`. Hence the
>    stationary profile "play a saddle action everywhere" is a pure-strategy
>    Markov-perfect equilibrium, and the pure-first hybrid solver never calls the
>    LP.
>
> 2. **Goal two or more cells wide.** At least one reachable `M_s` has
>    `maximin(M_s) < minimax(M_s)`: no pure saddle exists, an LP (or an
>    equivalent mixed-strategy solve) is genuinely required, and the equilibrium
>    at `s` is a non-degenerate mixed strategy for both players.
>
> 3. **Invariance.** The 0-vs-positive count of no-pure-saddle stage games is
>    unchanged across `gamma in {0.5, 0.9, 0.995}`, the 4- and 5-action move
>    sets (Littman's `STAND` included), the `win` and `rate` reward objectives,
>    and the exact undiscounted game solved by backward induction. It is also
>    unchanged across board size, board aspect ratio, and goal placement
>    (contiguous or with gaps) -- verified on boards up to `11x5` and `9x7`
>    ([generalize.md](generalize.md)).

### What is *not* claimed

- **Not a statement about stochastic transitions in general -- only about the
  random move order.** Add action-independent movement noise
  (`SoccerGame(slip=p)`: each player takes a random move with probability `p`)
  and mixed stage games appear even under deterministic resolution and even with
  a single goal cell. The goal-width switch is specific to Littman's move-order
  mechanism, where the unresolved coin only bites where the carrier has two
  lanes ([generalize.md](generalize.md) part B).

- Not that *every* multi-cell board has a mixed state for *all* `W, H` -- only
  that it does on every board in the tested grid, and that a multi-cell goal is
  *necessary* (never violated) for any mixed state to exist in this family.
- Not that every state on a multi-cell board is mixed -- the fraction is
  2.7-12% (a board-area dilution effect; [report.md](report.md) §RQ3).
- Not a closed-form geometric predicate for *which* states are mixed. A
  carrier-frame decision tree reaches precision 0.96 / recall 0.91
  (`scripts/geometry_model.py`); the deliverable is the per-state **certificate**
  (§3), not a formula.
- Not a board-size-free theorem. Direction 1 has a closed-form defender strategy
  and a dominance-solvability certificate per finite board
  ([proof.md](proof.md)); direction 2 is a finite exhaustive check with a
  per-stage-game certificate. Per the meeting feedback, a general proof is not a
  project deliverable.

---

## 2. Scope of the verification

`scripts/verify_mechanism.py` checks:

| part | configs | check |
|---|---|---|
| single cell | 5x3, 7x3, 5x5, 7x5 (goal `(H//2,)`) | 0 mixed stage games; cross-checked against `onecell.certify` (guard-strategy slack `0`, IEWDS failures `0`, undiscounted mixed `0`) |
| wider goal | 5x4 `(1,2)`, 6x5 `(1,2,3)`, 7x5 `(1,2,3)` | every mixed `M_s` gets a certificate that `certificate.verify` re-checks to `< 1e-6` |
| robustness | 5x3 vs 5x4, `gamma in {0.5,0.9,0.995}` x actions `{4,5}` x scoring `{win,rate}` + undiscounted | 0-vs-positive mixed count invariant (16/16) |
| perturbation | 5x4, `+/- 1e-6` noise on every `M_s` | no genuine mixed game (gap > 1e-4) and no sharp pure saddle (margin > 1e-5) changes label |

Observed on the 7x5 three-row goal: **94 / 2380** stage games are mixed, all 94
reachable from the kickoff, smallest pure-bound gap **3.8e-3** (≈ 3800x the
`1e-6` perturbation). The broader 127-config phase-diagram sweep
(`scripts/phase_diagram.py`) is the wide-coverage companion; this script is the
depth check with certificates.

---

## 3. The per-stage-game certificate

`soccer_nash/certificate.py` re-derives the pure/mixed label from the stage
matrix alone -- no LP, `O(A^2)` arithmetic -- and emits an object a reader can
check by hand.

**Pure** (`PureCertificate`): a cell `(i*, j*)` with `M[i*, j*]` the minimum of
its row and the maximum of its column. Then it is player 0's best reply to `j*`
and player 1's best reply to `i*` simultaneously, so `val(M_s) = M[i*, j*]`.
Carries the two slacks (both `0` for a true saddle).

**Mixed** (`MixedCertificate`):

- `maximin` and `minimax` with the full row-minimum / column-maximum vectors, so
  `max`/`min` are recomputable; `gap = minimax - maximin > 0` is the complete,
  LP-free proof that **no pure saddle exists**.
- a **best-reply cycle** -- e.g. `(row 3, col 2) -> (row 0, col 2) -> (row 0,
  col 1) -> (row 3, col 1) -> (row 3, col 2)` -- each leg an actual best reply,
  the walk closing on itself: the "someone always wants to deviate" witness.
- a **2x2 matching-pennies core**: two rows and two columns whose `2x2`
  submatrix `A` satisfies `A00 > A10`, `A11 > A01` (player 0 switches row with
  the column) and `A00 > A01`, `A11 > A10` (player 1 switches column with the
  row). Every one of the 94 (7x5) and 56 (5x4) mixed states has such a core --
  the interpretable "each player must guess" structure, the same shape as
  matching pennies and the near-goal soccer templates in
  [templates.md](templates.md).
- the residual **shape after iterated weak-dominance elimination** (never
  smaller than `2x2` on the mixed states), and its own `maximin < minimax` gap.
- the **exact equilibrium** `(p*, q*, v)` with `epsilon_equilibrium(M, p*, q*)`
  (`< 1e-15` in every case), the support sizes, and the mixing entropy in bits
  (`> 0` confirms genuine randomisation, not a pure strategy written as a
  degenerate mix).

`verify(cert, M)` recomputes every inequality and returns the worst violation;
across all tested boards it is at machine epsilon (`~1e-16`).

---

## 4. Why the switch happens (mechanism)

The certificate makes the informal argument in [mechanism.md](mechanism.md)
concrete:

- **One goal cell -> no guess.** With a single target `T`, every carrier route
  to a goal runs through `T`, and the defender has an unambiguous interposition:
  reach `T`'s row, then sit between the carrier and `T`. `onecell.guard_action`
  is that strategy in closed form; `onecell.certify` verifies it holds the
  carrier to `V*` from every state (guard slack `0`), which by weak duality
  forces `minimax(M_s) = V*(s)`, and iterated weak dominance closes the other
  half. There is only one cell worth contesting, so the random resolution order
  changes *timing* but not *which* cell each player wants -- each has a
  direction that is a best response regardless of the other. Pure saddle.

- **Two goal cells + a defender one move away -> a genuine guess.** The carrier
  near the mouth threatens both cells; the defender can cover one lane or hold
  the forward cell, not both. The carrier's best lane now depends on the
  defender's cover and vice versa -- the best replies cross (the certificate's
  cycle and `2x2` core). No direction is a best response to both defender
  choices: mixing is forced.

- **The random order is the enabling condition, not the goal width alone.**
  Under deterministic "carrier wins contests" resolution, or a fair coin flip on
  contested squares, *every* stage game on *every* tested board -- one cell or
  many -- has a pure saddle, and the deterministic game has a constructive
  memoryless equilibrium ([proof.md](proof.md)). It is the random *order* --
  which couples the ball-steal outcome to *both* players' concurrent choices --
  that turns the two-lane geometry into a matching-pennies stage game.

- **What is common to both.** A stage game needs mixing when the transition puts
  the outcome on a coin neither player controls. The random move order supplies
  that coin only where the carrier has two lanes and the defender is between
  them, so mixing tracks the goal width. Action-independent movement noise
  (`slip`) supplies the same coin on every square, so under `slip` mixing
  appears at any goal width ([generalize.md](generalize.md)). Reading the `2x2`
  cores of the `7x5` mixed states: the carrier's crossing pair is almost always
  a vertical move -- it is picking which goal row to head for -- and the
  defender's pair mirrors it.

---

## 5. Reproduction

```
make verify        # scripts/verify_mechanism.py -> experiments/mechanism_certificate.json
                   #                                 docs/figures/gallery/mechanism.svg
make generalize    # scripts/generalize.py: board scale, goal shape, slip, mechanism
make proof         # scripts/onecell_proof.py: the single-cell closed-form certificate
make phase         # scripts/phase_diagram.py: the 127-config wide sweep (~8 min)
```

`experiments/mechanism_certificate.json` holds the full per-part results plus one
fully expanded worked example (`worked_example`: the state, the mixed
certificate with its cycle, core, and exact equilibrium). Tests:
`tests/test_certificate.py`, `tests/test_verify_mechanism.py`.

---

## 6. Position relative to prior work

**Littman (1994), "Markov games as a framework for multi-agent RL."** Introduces
the discrete soccer game *specifically* as an example where the minimax-Q
optimal policy is probabilistic, and exhibits **one** such state (his Figure 2:
the carrier mixes 50/50 on a near-goal square). What Littman does not do: say
*which* states mix, or *how many*, or connect it to the goal geometry, or
separate the role of the stochastic move-resolution rule from the role of
simultaneity. This result is the machine-checked map -- the exhaustive
pure-vs-mixed classification over the reachable state space, with an
independently checkable certificate per state, and the goal-width /
resolution-rule switch that turns the mixed region on and off. Littman's Figure 2
state is reproduced exactly (`scripts/littman.py`, `make littman`).

**Shapley (1953); zero-sum stochastic games; the ordered-field property**
(Parthasarathy-Raghavan; Filar-Vrieze). The value and optimal stationary
strategies of a discounted zero-sum stochastic game lie in the same ordered
field as the data. Here that means the mixed probabilities are rational
functions of `gamma` and the payoffs, so the equilibria the certificate reports
are *exact* for rational data -- the `epsilon_equilibrium < 1e-15` residual is
floating-point noise around an exact rational solution, not an approximation
error. The "one target vs. many" contrast is the discrete-soccer instance of the
classical observation that single-target pursuit collapses to a race while
multi-target pursuit need not.

**Concurrent reachability games** (de Alfaro-Henzinger-Kupferman; Everett
recursive games). Soccer is a concurrent (simultaneous-move) reachability game;
in general such games need randomised strategies and admit only limit / epsilon
values. This result *refines* that picture for discrete soccer: simultaneity
alone is not enough to force mixing here -- with deterministic or coin-flip
contest resolution the game is positionally determined and has a pure memoryless
equilibrium at every goal width. It is the stochastic *tie-break coupled to both
actions* that produces genuine matching-pennies stage games, and only when the
goal gives the carrier two lanes.

**Pursuit-evasion / graph pursuit games.** Cops-and-robbers-style pursuit on
graphs is deterministic and positional; the differential-games literature
(Isaacs) finds mixed / singular strategies precisely near the target set of a
pursuit problem, which is where the soccer mixed region also concentrates (41%
of the equilibrium-path occupancy sits on 3.95% of states -- `scripts/occupancy.py`).

**What is new here, stated conservatively.** To our knowledge the switch from a
pure-everywhere equilibrium to a provably mixed one as a function of goal width
-- with the stochastic move-resolution rule as the necessary enabling condition
-- has not been isolated for the discrete grid-soccer Markov game in this form.
The contribution is the exhaustive certificate-backed classification and the
identification of the switch, not a new solution concept or a general theorem.
