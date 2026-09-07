# Environment assumptions and A10 fidelity

The headline result -- "every stage game admits a pure saddle" -- is only as
strong as the environment it is measured on. This page states exactly what the
implemented environment does, marks every place the A10 text is *interpreted*
rather than quoted, and lists what to confirm with course staff.

## Quoted directly from the A10 page

- 7x5 grid; state `(x0, y0, x1, y1, b)`; actions `U D L R`, chosen
  simultaneously.
- "If the two players try to occupy the same square, only the player with the
  ball will move to that square, and the other player will get the ball."
- "If the two players try to swap squares, they will swap, and the other player
  will get the ball."
- "The game ends in a tie if no one scores in 100 steps."
- Reward: `+1` win / `-1` loss / `0` otherwise, no discounting, for the Part 1
  evaluation.

## Interpreted (not quoted)

These are choices consistent with the two stated collision rules but not
spelled out. **Each is a place the result could shift if course staff intended
something different.**

| sub-case | what the A10 text says | this repo's reading |
|---|---|---|
| start positions, goal rows | parameterised by student ID, redacted on the public page | Littman geometry: player 0 left, player 1 right; goal mouth on rows `{1, 2, 3}`; `initial_state` = centre row, opposite ends. Configurable via `SoccerGame(width=, height=, goal_rows=)`. |
| carrier moves into a *stationary* opponent (not a shared empty square) | nothing explicit | treated as "trying to occupy the same square": the carrier is blocked, stays put, and possession passes to the stationary player (`_resolve_with_winner`, `t0 == t1 == pos[loser]` branch). |
| non-carrier moves into the carrier | nothing explicit | same rule: the non-carrier is blocked and *takes* the ball (a bump = steal). |
| out-of-bounds move | nothing explicit | clamped -- the player stays. Exception: a carrier moving into the opponent's goal edge on a goal row scores. |
| both players move to the same *empty* square | "only the player with the ball will move to that square" | the carrier takes the square; the non-carrier stays and gets the ball. |
| move into a square the opponent is vacating (opponent moves elsewhere, no shared target) | nothing explicit | allowed -- both moves succeed, possession unchanged. |

The deterministic rule is `_resolve_with_winner(..., winner = b)` -- the carrier
wins every contest. The `random` and `coinflip` variants change only *who* wins
the contest, nothing else.

## What to confirm with course staff

> Are the deterministic collision semantics -- specifically (a)
> carrier-vs-stationary-opponent, (b) non-carrier bump = steal, and (c) swap
> possession -- exactly the intended A10 semantics? And what are the
> ID-specific start position and goal rows?

If (a)-(c) differ from the intended rule, the "pure saddle at every state"
result must be re-measured; the solver and analysis are unaffected.

## Three claims, kept separate

The experiments in this repo establish different things:

- **Claim A (established).** For the implemented deterministic transition model,
  every one of the 2380 enumerated non-terminal stage games -- built from the
  converged value function -- has a pure saddle point
  (`maximin == minimax`).
- **Claim B (strongly supported).** The Markov game therefore has a *stationary
  pure-strategy equilibrium*: playing a pure saddle action at every state is a
  best response to itself. Value iteration converges to it and `hybrid` and
  `mixed` return the same value function.
- **Claim C (not established here).** The *theoretical* A10 game -- with the
  exact intended semantics, over all reward and discount settings -- necessarily
  admits a pure-strategy equilibrium. This would need a proof, not enumeration,
  and depends on resolving the interpretation above.

The report makes Claim A and B; it does not make Claim C.
