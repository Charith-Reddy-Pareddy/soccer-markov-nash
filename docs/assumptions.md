# The A10 geometry instance, and its interpreted rules

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
