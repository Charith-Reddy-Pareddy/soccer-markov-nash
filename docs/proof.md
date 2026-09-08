# Pure equilibria of the soccer game

`soccer_nash/attractor.py`, `soccer_nash/onecell.py`, `soccer_nash/dominance.py`.

Two results, from strongest to weakest:

1. **The deterministic game has an explicit pure memoryless equilibrium**
   (constructive, any goal width). See "The deterministic game" below.
2. **The random-move-order game with one goal cell has a pure saddle at every
   state** -- verified per finite board three ways, with the defender's optimal
   strategy in closed form. This is [assumptions.md](assumptions.md)'s Claim C in
   the one case within reach; a board-size-free proof of the carrier's half is
   open (stated at the end).

---

## The deterministic game: a constructive equilibrium

For each player, `win_attractor(game, player)` computes the set of states from
which that player *forces* a goal against any opponent play -- the concurrent
controllable-predecessor least fixpoint -- with a rank (minimum forced-win
length). `positional_profile` then gives a **pure memoryless strategy per
player**: the rank-reducing attractor move on the player's own winning set, and
a *safety* move (one that keeps the game out of the opponent's attractor for
every opponent reply) everywhere else.

`verify_positional_equilibrium` checks that this profile realizes the exact
undiscounted value `V*` at every state -- **0 mismatches** for every board
`w x h` with `w in {3,5,7,9}`, `h in {3,5,7}`, `w*h <= 45`, at goal widths 1 and
3 (`scripts/positional.py`). The win attractors coincide exactly with the
`V* = +1` / `V* = -1` sets, the complement is the `V* = 0` draw region, and both
strategies are memoryless.

So the deterministic soccer game is **positionally determined**: a pure
memoryless Markov-perfect equilibrium exists and is given in closed form. (This
is why the *random* move order, not the goal width, is what forces mixing in the
first place -- &sect;RQ1.)

---

## The random single-cell game

The rest of this page is the harder claim: the *random*-move-order game with one
goal cell. It is **not yet a proof for arbitrary board dimensions**; it is a
proof for every finite board (a finite check), plus a closed-form optimal
strategy for the defender and a hand argument for the easy regions.

## Statement

> **Theorem.** For the random-move-order soccer game (Littman resolution) with
> exactly one goal cell per side, every stage game has a pure-strategy saddle
> point: `maximin(M_s) = minimax(M_s) = V*(s)`. Hence a pure equilibrium exists
> and the pure-first hybrid solver never calls the LP. The game is a **forced
> draw** — `V*(kickoff) = 0` — under every resolution rule.

Verified exactly (`scripts/onecell_proof.py`) for every board `w × h` with
`w ∈ {3,5,7,9,11}`, `h ∈ {3,5,7}`, `w·h ≤ 55`, three ways:

1. **stationary**, `γ ∈ {0.5, 0.7, 0.9, 0.95, 0.99}` — guard slack `0`, IEWDS
   failures `0`;
2. **exact undiscounted** (100-step backward induction, `run_finite_horizon`) —
   `0` mixed stage games across the whole non-stationary horizon;
3. and `V*(kickoff) = 0` in every case.

0 exceptions.

## Notation

Fix the single goal row `g`. The carrier is the ball holder; it attacks the
goal cell `T = (W−1, g)` if it is player 0, `T = (0, g)` if player 1. The
defender is the other player. `M_s[a₀, a₁] = E_order[r₀ + γ V*(s′)]` is the
stage game (player 0 maximises, player 1 minimises). Two standing facts:

- **Weak duality.** `maximin(M) ≤ val(M) ≤ minimax(M)` for any matrix game.
- **Shapley.** `val(M_s) = V*(s)` at the fixed point (in mixed strategies).

So a pure saddle at `s` is equivalent to *either* `minimax(M_s) ≤ V*(s)` *or*
`maximin(M_s) ≥ V*(s)` being an equality — the other follows from duality and
Shapley, and then `maximin = minimax`.

## Part 1 — the defender's closed-form optimal strategy

**Guard strategy `β(s)`** (`onecell.guard_action`): the defender

1. if not on row `g`, steps vertically toward `g`;
2. once on row `g`, steps horizontally toward `T` (clamped at the wall);
3. never steps onto the carrier's current cell.

**Lemma 1.** Playing `β`, the defender holds the carrier to at most `V*(s)`
from every state: for every carrier action `a`,
`E_order[r₀ + γV*(s′)] ≤ V*(s)` when player 0 carries, and `≥ V*(s)` when
player 1 carries.

*Verification.* `onecell.certify` evaluates all four carrier replies to `β` at
every state and reports the largest violation ("guard slack"). It is `0` (to
machine precision) for every board and `γ` in the grid above.

*Consequence.* For a state where player 0 carries, Lemma 1 gives
`minimax(M_s) ≤ max_a M_s[a, β(s)] ≤ V*(s)`; with weak duality and Shapley,
`minimax(M_s) = V*(s)`. For a state where player 1 carries, symmetrically
`maximin(M_s) = V*(s)`.

*Why it works — the interpretable content.* With a single goal cell the
defender's job is unambiguous: get between the carrier and the one cell `T`,
then sit on the one row that leads to it. Every carrier route to a goal runs
through `T`, so a defender on row `g` at a column `≥` the carrier's column is
never wrong-footed — there is no second lane to be beaten on. The carrier
gains nothing by moving off row `g` (that strictly lengthens its only path to
`T`) and nothing by charging the defender (under the Littman rule the carrier
*loses the ball* by moving into an occupied cell). This is exactly the argument
in [mechanism.md](mechanism.md), now backed by an explicit strategy.

## Part 2 — every stage game is dominance-solvable

**Lemma 2.** For the single-cell game, every stage game `M_s` is solvable by
iterated elimination of weakly dominated actions (`dominance.iewds`): the
residual is a `1×k`, `k×1`, or pure-saddle game.

*Verification.* `dominance.is_iewds_solvable` on every stage game — 0 failures
across the grid.

Since an IEWDS-solvable game has a pure saddle, Lemma 2 **alone proves the
theorem for each finite board**. Part 1 is not logically necessary, but it adds
the defender's strategy in closed form and an independent proof of one half.

*Hand proof of the easy regions* (these are the bulk of the states):

- **Scoring states.** If the carrier is on `T` with the ball, its scoring
  action wins outright regardless of the defender (the move is off the board;
  nothing can block it). It strictly dominates; the residual is one row,
  `V* = ±1`.
- **The draw region.** `scripts/onecell_proof.py` reports `V(kickoff) = 0` for
  every board, and enumeration shows `V* = 0` on a large set `Z` (about half the
  states: 1070 of 2380 on 7×5). About half of the stage games on `Z` are the
  all-zero game (trivially a pure saddle at `0`); the rest are small mixed-sign
  matrices where the defender's guard column still holds the carrier to `0`
  (Lemma 1) and weak-dominance elimination removes the carrier's losing rows to
  leave one. None require mixing.
- **The clean-race region.** Where one player can force a goal in `k` moves
  against any defence, that player's distance-reducing move weakly dominates
  after eliminating the opponent's hopeless replies, and `V* = ±γ^{k}` (or
  `±γ^{k}/2` where the resolution coin decides the last step — still a pure
  saddle at each stage).

## Why ≥ 2 goal cells breaks both parts

- **Guard becomes a guess.** With goal rows `g₁ < g₂`, step 1 of `β` —
  "go to row `g`" — is ambiguous. A carrier on `g₁` with the defender between
  it and `T₁` can drop to `g₂`… except `g₂` is now *also* a scoring row, so the
  defender must have committed to `g₁` or `g₂` and can be wrong. That
  commitment is the matching-pennies guess.
- **IEWDS stalls.** On the 94 mixed states of the 7×5 three-row-goal game,
  weak-dominance elimination leaves a `2×2` (or larger) residual with **no**
  dominated action — exactly the templates in [templates.md](templates.md).
  `dominance.iewds` residual sizes on those 94 states: `3×3` (30), `3×4` (16),
  `4×3` (16), `3×2` (14), `2×3` (14), `2×2` (4) — never smaller.

## The remaining gap

Part 2 proves the theorem per board but is a finite check, not an argument for
all `W, H`. Part 1 proves the **defender's** half (`minimax = V*`) with a
strategy that visibly generalises, but the **carrier's** half
(`maximin = V*` — the carrier has a *pure* action securing the value) rests on
Lemma 2's elimination, not on a closed-form carrier strategy. A greedy
"race to `T`, sidestep the defender" carrier rule was tried and fails on ~5% of
states (small-value states where securing the edge needs a non-obvious move).

Closing the gap means either (a) a carrier strategy mirroring `β` in
generality, or (b) showing weak-dominance elimination always terminates in a
pure profile for the single-cell transition structure — an induction on the
carrier's distance to `T`, with the wall-clamping and swap cases handled
explicitly.
