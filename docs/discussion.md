# Discussion and open questions

Questions the project raised, with the best answer its evidence supports and
what would settle each. These are research-level, not assignment deliverables.
For how the meeting feedback maps to the repo, see [advisor.md](advisor.md).

---

## 1. Can the single-cell pure-saddle theorem be finished?

**What is established.** For a one-cell goal per side, under Littman's random
move order, every stage game has a pure saddle -- verified three independent
ways ([proof.md](proof.md)): the defender's closed-form guard strategy secures
`V*` at every state (so `minimax(M_s) = V*(s)` by weak duality); every
stationary stage game is solvable by iterated weak-dominance elimination; and
the exact *undiscounted* game (100-step backward induction) has zero mixed stage
games across its whole non-stationary horizon. `V*(kickoff) = 0` -- a forced
draw. The gap is a **board-size-free** argument (the above are finite checks
plus one player's strategy).

**What is now done.** For the **deterministic** game this is *solved*
constructively (`soccer_nash/attractor.py`, [proof.md](proof.md)): each player's
win attractor -- the concurrent controllable-predecessor fixpoint -- coincides
exactly with the `V* = ±1` sets, and the pure memoryless profile "attractor move
on your winning set, safety move elsewhere" realizes `V*` at every state (0
mismatches, all boards tested, goal widths 1 and 3). The deterministic game is
positionally determined.

**What remains: the *random* single-cell game.** It is a stochastic reachability
/ safety game -- player 0 wants the ball to reach the one cell `T`, player 1
wants to prevent it forever. With a **single** target the concurrency that makes
matching pennies collapses: on the one approach to `T`, a carrier that moves
into the defender *loses the ball* -- a strict loss -- so the carrier never
contests and the defender never needs to. The naive "race to `T`, sidestep"
carrier rule failed on ~5% of states precisely because it is not a value-aware
strategy: near the draw boundary the value-preserving carrier move is sometimes
"wait" or "retreat". The clean object is a *stochastic* attractor (an MDP-style
`P(reach T)` fixpoint) with a pure threshold strategy; showing it is optimal --
by induction on the distance to `T`, with the ball-loss-on-contact rule as the
base case -- would close the gap board-size-free. Related reading: concurrent
reachability positional determinacy (de Alfaro-Henzinger-Kupferman); Filar &
Vrieze, *Competitive Markov Decision Processes*.

---

## 2. Is "one goal cell ⟹ pure, two or more ⟹ mixed" a known result?

**What is established.** The phase diagram (127 board/goal-width configs,
`--analyze`) shows goal-mouth width is a **binary switch** for whether *any*
state needs mixing -- perfectly, independent of board size -- and the *fraction*
that mix is a board-area dilution effect, not a goal-width effect. Verified
undiscounted and at every `γ` tested.

**Best answer.** This looks like the game-theory distinction between a
**single-target** and a **multi-target** pursuit game. One target: the evader
(defender) has a dominant "guard the target" strategy, so the game is
positionally determined even when concurrent. Two or more adjacent targets: the
pursuer (carrier) can split its threat between them, the defender must commit to
covering one, and under `½`-`½` resolution order that commitment is a genuine
guess -- a matching-pennies stage game. It is adjacent to, but not the same as,
the **orderfield property** of stochastic games (whether the value lies in the
same ordered field as the data): single-controller and switching-controller
games have it, general concurrent games need not, and the single-cell soccer
game trivially has it because its undiscounted value is in `{-1, 0, +1}`.

I am not aware of this stated as "a one-cell goal makes discrete soccer
pure-solvable". Littman's 1994 paper introduces soccer *because* it needs mixing
"in the place where they had to have it" -- our contribution is locating and
characterizing that place. Worth a literature check: recursive games (Everett),
the orderfield-property literature (Parthasarathy-Raghavan, Filar), and
pursuit-evasion on graphs.

---

## 3. Move *order* vs. randomness -- what actually creates the mixing?

**What is established, and answered.** A fair-coin tie-break (`coinflip` -- a
coin, not possession, decides who wins a contested square or a swap) gives a
value function **bit-identical** to the deterministic game (`max |V_det -
V_coin| = 0` at every `γ`, discounted and not). Littman's random *move order*
(apply the two moves in a uniformly random sequence) does force mixing.

**The distinction.** Stochasticity that is **independent of the joint action** --
a tie-break coin flipped only when a contest is already unavoidable -- cannot
create a mixed equilibrium in a game that has pure saddles under the
deterministic limit: optimal play simply never enters a contest it might lose,
so the coin is never flipped on the equilibrium path. Stochasticity that is
**coupled to the joint action** -- the move order, which decides whether a move
"into the other player's current cell" succeeds, and that depends on *both*
players' targets at once -- can, because it makes the stage-game payoff itself a
function of the action profile in a matching-pennies pattern. This is a clean
one-line characterization and the project verifies both halves.

---

## 4. Why does freeze-then-iterate (policy iteration) thrash?

**What is established.** `run_policy_iteration` reaches the same fixed point as
value iteration with ~5x fewer LP solves, but the frozen stage strategies stay
*maximally stale* -- a full pure-strategy flip from the current Nash -- for 16
outer rounds, then snap to `< 1e-8` (`staleness_trace`).

**Best answer.** Turn-based (perfect-information) stochastic games have a
monotone strategy-improvement algorithm (Hoffman-Karp, Condon) with clean
convergence. **Concurrent** stochastic games do not: the value can fail to be
attained by stationary strategies at all (Everett's recursive games), and where
it is attained, an improvement step need not be monotone. The 16-round staleness
is the expected shape of that: freezing strategies from a not-yet-converged `Q`
gives a stage saddle that is a pure flip away from the true one; the linear
evaluation sweeps then propagate the wrong continuation values until `Q` crosses
the threshold where the frozen saddle flips, and everything corrects at once.
It is not a bug -- it is why plain value iteration with a per-sweep Nash cache
wins on a game this size, and why freeze-then-iterate only pays off where the
equilibrium *solve* (not the matrix construction) dominates the cost.

---

## 5. What the mixed-strategy region depends on

Pulling the sweeps together -- **which parameters change whether/where a stage
game needs a mixed Nash equilibrium, and which do not.**

| parameter | effect on the no-pure-saddle region | evidence |
|---|---|---|
| **move-resolution rule** | **determines it.** deterministic / coinflip / `blend ≤ 0.5` → **0** mixed; random / `blend > 0.5` → mixing, with a sharp threshold and an overshoot | [blend.md](blend.md), §3 |
| **goal-mouth width** | **binary switch.** 1 cell → 0 mixed on every board; ≥ 2 → mixing on every board | phase diagram, §2 |
| board size | scales the *fraction* only (area dilution, R² 0.64); does not create or remove mixing | phase diagram |
| discount `γ` | shifts *which* states mix by a few percent (94 at 0.9, 122 at 0.5, 102 at 0.995); the region is otherwise stable | RQ4 |
| **kickoff position** | **no effect** on the region; only `V(kickoff)` moves | RQ1 |
| **reward objective** (`win` vs `rate`) | **no effect** -- the identical 94 states; only the value *range* changes | [reward.md](reward.md) |
| the `stand` action | barely -- 48 of 56 no-saddle states shared with the 4-action game; changes the *content* of the mix, not its location | [littman.md](littman.md) |
| the training opponent (learning) | none for an exact solver: minimax-Q's update is opponent-independent, so "trained vs random" and "trained vs self" give the same Nash policy | [tournament.md](tournament.md) |

The pattern: a mixed Nash equilibrium is a property of the **stage game
`M(s)`**, and only the two parameters that reshape `M(s)`'s *strategic* structure
-- the resolution rule (which couples the payoff to both actions) and the goal
width (which gives the carrier two threats) -- move the region. Parameters that
only rescale or shift `M(s)` uniformly (the objective, the kickoff, the
discount) leave it where it is. This is the concrete form of "mixed NEs do not
depend on the transition function / the horizon": they depend only on whether
the *stage game* is a matching-pennies game.

Occupancy ([occupancy.md](occupancy.md)) adds the practical footnote: the mixed
states are 4% of the space but 41% of the equilibrium path.

---

## 6. Does the mixed policy actually matter, or is it a technicality?

It is the whole game. Reproducing Littman's Table 3 exactly
([tournament.md](tournament.md)): a **greedy** policy that commits to a pure
action at every state beats a random opponent by `+0.92` goal difference but
*loses* `−0.59` to a challenger trained against it — from dominant to beaten.
Littman's own **hand-built** policy (which beats a random opponent ~76% of the
time) collapses the same way: `+0.72 → −0.68`. The **minimax** policy beats the
weak opponents (`+0.67` vs. random, `+0.28` vs. hand-built) and still wins
`+0.15` against its worst case. Of the four policies tested, minimax is the only
one that both presses an advantage and survives a strong opponent.

The reason is the matching-pennies structure of the mixed states: any
deterministic policy plays one row of each, and the challenger plays the column
that beats it, every time — "every deterministic offense has a perfect defense"
(Littman).
Solving the LP on the 3.95% of states that need it is not a numerical nicety;
it is the difference between a policy that can be exploited to a loss and one
that cannot be exploited at all.

## How the experiments were run

See [methods.md](methods.md) for the solver, the state encoding, seeds, repeat
counts, board ranges, and how to reproduce every table.
