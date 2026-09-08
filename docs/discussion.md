# Discussion and open questions

Four questions the project raised, with the best answer its evidence supports
and what would settle each. These are research-level, not assignment
deliverables.

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

**Best answer.** The single-cell game is a **reachability / safety game**:
player 0 wants the ball to reach the one cell `T` (reachability), player 1 wants
to prevent that forever (safety). Deterministic reachability games on finite
graphs are *positionally determined* -- both players have optimal pure
positional (memoryless) strategies, computed by an attractor fixpoint. Soccer is
*concurrent* (simultaneous moves), which in general breaks positional
determinacy (matching pennies is a concurrent reachability game with no pure
value). But with a **single** target the concurrency collapses: on the one
approach to `T`, a carrier that moves into the defender *loses the ball* -- a
strict loss -- so the carrier never contests and the defender never needs to.
The stage game degenerates to a turn-order-independent one, and the attractor
strategy is optimal for both. The naive "race to `T`, sidestep" carrier rule
failed on ~5% of states precisely because it is not the attractor strategy: near
the draw boundary the value-preserving carrier move is sometimes "wait" or
"retreat", not "advance".

**To close it.** Prove the attractor strategy achieves `V*` for the single-cell
transition structure -- an induction on the attractor rank, with the
wall-clamping and the ball-loss-on-contact rule handled as the base cases. This
is a cleaner object than the weak-dominance elimination and should generalize to
any board. Related reading: reachability-game positional determinacy;
de Alfaro-Henzinger-Kupferman on concurrent reachability; Filar & Vrieze,
*Competitive Markov Decision Processes*.

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

## How the experiments were run

See [methods.md](methods.md) for the solver, the state encoding, seeds, repeat
counts, board ranges, and how to reproduce every table.
