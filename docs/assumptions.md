# The A10 geometry instance, and its interpreted rules

The assignment itself:
[CS 540 A10](https://pages.cs.wisc.edu/~yw/CS540S26A10.html). Quoted text
below is copied verbatim from that page.

The game family and the design rationale are in [design.md](design.md). This
page is about one instance -- the CS 540 A10 geometry, `A10SoccerGame` -- which
is where the "every stage game admits a pure saddle" result is measured. The
headline result is only as strong as that environment, so this page marks every
place the A10 text is *interpreted* rather than quoted, and what each
interpretation would change.

(These interpretations are also just transition-design choices; the family
supports others. Nothing here blocks the research direction -- it is bookkeeping
for the one instance that carries a numeric claim.)

## Quoted directly from the A10 page

- 7x5 grid; state `(x0, y0, x1, y1, b)`; actions `U D L R`, chosen
  simultaneously.
- "If the two players try to occupy the same square, only the player with the
  ball will move to that square, and the other player will get the ball."
- "If the two players try to swap squares, they will swap, and the other player
  will get the ball."
- "The game ends in a tie if no one scores in 100 steps."
- Reward: `+1` win / `-1` loss / `0` otherwise, **no discounting** -- that is the
  game. (The page allows a self-designed discounted *training* reward, but that
  is a training choice, not the game's definition.) The solver uses `gamma < 1`
  purely as a contraction device and checks the results across `gamma`.
- Per this netID (shown after entering it on the page): player 0 (left) starts
  at `[0, 1]`, player 1 (right) at `[6, 3]`, ball with player 0.

## Interpreted (not quoted)

These are choices consistent with the two stated collision rules but not
spelled out. **Each is a place the result could shift if course staff intended
something different.**

| sub-case | what the A10 text says | this repo's reading |
|---|---|---|
| start positions | per netID: after entering the netID the A10 page shows player 0 at `[0, 1]`, player 1 at `[6, 3]` | used verbatim -- `A10SoccerGame().initial_state() = (0, 1, 6, 3, 0)`. A different netID gets different positions; pass `p0_start=` / `p1_start=`. |
| goal rows / how a score triggers | not stated on the page | `goal_rows = {1, 2, 3}` (middle three of the 5); a carrier on a goal row that moves off its attacking edge scores. This is **load-bearing**: a one-row goal is never mixed, `>= 2` always is. Confirm with staff. |
| carrier moves into a *stationary* opponent (not a shared empty square) | nothing explicit | treated as "trying to occupy the same square": the carrier is blocked, stays put, and possession passes to the stationary player (`_resolve_with_winner`, `t0 == t1 == pos[loser]` branch). |
| non-carrier moves into the carrier | nothing explicit | same rule: the non-carrier is blocked and *takes* the ball (a bump = steal). |
| out-of-bounds move | nothing explicit | clamped -- the player stays. Exception: a carrier moving into the opponent's goal edge on a goal row scores. |
| both players move to the same *empty* square | "only the player with the ball will move to that square" | the carrier takes the square; the non-carrier stays and gets the ball. |
| move into a square the opponent is vacating (opponent moves elsewhere, no shared target) | nothing explicit | allowed -- both moves succeed, possession unchanged. |

The deterministic rule is `_resolve_with_winner(..., winner = b)` -- the carrier
wins every contest. The `random` and `coinflip` variants change only *who* wins
the contest, nothing else.

## If the A10 semantics were meant differently

The interpreted sub-cases (a) carrier-vs-stationary-opponent, (b) non-carrier
bump = steal, (c) swap possession, and the goal rows, are choices consistent
with the two quoted rules. If the intended rule differs, the "pure saddle at
every state" number must be re-measured on that variant; the solver, the
visualization, and the analysis are unaffected. The goal-row choice is the only
load-bearing one -- a one-row goal is never mixed, `>= 2` always is -- and the
phase diagram sweeps it explicitly, so the qualitative result holds either way.

## Three claims, kept separate

The experiments in this repo establish different things:

- **Claim A (established).** For the implemented deterministic transition model,
  every stage game has a pure saddle point (`maximin == minimax`). Checked two
  ways: exact 100-step backward induction on the *undiscounted* game (all
  238 000 (state x step) stage games, `experiments/undiscounted.csv`), and the
  stationary `gamma < 1` value iteration at every `gamma` from 0.5 to 0.995
  (all 2380 states).
- **Claim B (established, constructive).** The deterministic Markov game has an
  explicit **pure memoryless** equilibrium: each player's win-attractor strategy
  on its forced-win set, a safety move elsewhere (`soccer_nash/attractor.py`,
  `scripts/positional.py`). Verified to realize `V*` at every state, all boards
  tested, goal widths 1 and 3 -- the deterministic game is positionally
  determined.
- **Claim C (not established here).** The *theoretical* A10 game -- with the
  exact intended semantics, over all reward and discount settings -- necessarily
  admits a pure-strategy equilibrium. This would need a proof, not enumeration,
  and depends on resolving the interpretation above.
- **Claim C', single goal cell (partly established).** For the random-resolution
  game with *one* goal cell per side, every stage game has a pure saddle. The
  defender's optimal strategy is closed-form and verified optimal
  ([proof.md](proof.md), Part 1); dominance-solvability is machine-checked for
  every board up to 11x5 and every discount 0.5-0.99 (Part 2). A board-size-free
  proof of the carrier's half is still open.

The report makes Claim A and B, and Claim C' for the single-cell case; it does
not make the general Claim C.

## Solver and learner assumptions (beyond the game rules)

The tables above cover the *game*. These are the choices the *solvers and
learners* make on top of it. "Mine" means nothing in the A10 text or the group
notes asked for it.

| # | assumption | where | status |
|---|---|---|---|
| S1 | Value iteration is **stationary** with `gamma = 0.9`; the state has no step counter, so the 100-step tie is not modelled. The undiscounted game is solved separately by backward induction (`run_finite_horizon`). | `nash_q.py` | mine (A10 allows a discounted training reward) |
| S2 | Each stage game is solved as a **zero-sum matrix game** (minimax value, Shapley 1953). | `nash_q.py` | standard |
| S3 | Where a stage game has several equilibria (degenerate states), the reported policy is **one** of them (the LP vertex); the value and Q are unique, the policy is not. | `degeneracy.csv` | mine |
| S4 | Pure-vs-mixed is decided with a tolerance that scales with the matrix entries. | `numerics.py` | mine |
| S5 | **Policy gradient (REINFORCE / A2C / PPO)** trains on the same `+/-1` goal reward, `gamma = 0.9`, from the A10 kickoff; episodes restart at kickoff after a goal. Off-path states are therefore visited rarely, yet the evaluation scores *all* 2,380 states against the exact solve. | `policy_gradient.py`, `actor_critic.py` | mine |
| S6 | The 100-step tie is **not** enforced inside PG episodes; a rollout is cut every 100 steps and the cut is bootstrapped by the critic (A2C/PPO) or given zero continuation (REINFORCE). | same | mine |
| S7 | The two players have **separate** policy and value networks (no mirror constraint). A shared-weights variant is tested in `policy_gradient.md`; mirror-symmetry is *measured*, not imposed. | same | mine |
| S8 | A2C/PPO hyperparameters (`lr 1e-3`, hidden 64, entropy 0.01, GAE `lambda 0.95`, clip 0.2, 4 PPO epochs, 2000 x 100 steps) are **untuned**, and match REINFORCE's budget. A loss for PPO/A2C is therefore not evidence about the algorithm. | `actor_critic.py` | mine |
| S9 | "Wins" are reported as **expected discounted goal difference** from kickoff against a random policy and against the exact best response, not as win probability. A tie is worth 0. | `evaluate_policy_gradient` | mine |
| S10 | **Fictitious play** is run on the exact stage matrices (fixed `V*`), both players updating simultaneously, ties broken by lowest action index, first action `U`. It is *not* yet run as a learning dynamic inside the Markov game. Run on the **random move-order** board, where 94 stage games are genuinely mixed; the deterministic A10 board has none, so FP is trivial there. | `fictitious_play.py` | mine |
| S11 | The **dog game** is *not implemented*. Its dynamics, speeds, capture rule, reward and horizon are not specified anywhere I can see. | -- | blocked on the professor |

## Questions for the professor

1. **Dog game definition.** Pursuit-evasion in the plane? Who is the dog and who
   the sheep, what are the two speeds, the arena, the capture radius or goal,
   the reward, and the horizon? Is time discrete? Is the angle-radius output a
   displacement `r(cos t, sin t)` with `r <= delta` for *both* players?
2. **Policy distribution for the angle and radius.** A von Mises for `t` and a
   Beta (or squashed Gaussian) for `r`, or a deterministic output plus noise?
3. **Which soccer variant is the target for PG and fictitious play?** The A10
   deterministic board has **0** mixed stage games, so there is nothing mixed
   to learn; the random move-order board has 94. Should the learners be judged
   on the deterministic A10 game, the random one, or both?
4. **Discount.** Is training at `gamma = 0.9` acceptable, or should the learners
   see the undiscounted game with a 100-step timer (a non-stationary policy that
   needs the step number as an input)?
5. **"Win rate".** For the random / NE / best-response comparison: is it
   `P(score first within 100 steps)`, and how is a tie counted? Against the exact
   NE policy the expected goal difference is just `V*`, so a win rate is the more
   informative number there. The note's "NE ??? 1/2" looks like it asks exactly this.
6. **"Check symmetry."** Impose mirror equivariance (one network, a side flag),
   or just test whether the two separately trained policies are mirror images?
7. **Fictitious play scope.** Stage games with the exact `V*` (done), or FP as a
   full learning dynamic in the Markov game, with a learned best-response
   network per state?
8. **`Q*` in "DQN, PG -> soccer => Q*".** Is that the per-state 4x4 joint-action
   matrix `Q(s, a0, a1)` the exact solver produces? That is what is compared.
9. **Continuous best response (page 3).** Bisection on `Q'`, finite-difference
   gradient, quadratic/LQR approximation: are these for the dog game now, or later?
