# Questions for course staff

The "every stage game has a pure saddle" result is measured on a specific
transition function. A10 states two collision rules but leaves several sub-cases
open; this repo resolves them one way (see `docs/assumptions.md`). If the
intended rule differs, **only the enumeration has to be re-run** -- the solver
and the analysis are unchanged.

## A. The redacted per-student parameters

The public A10 page parameterises these by student ID and redacts them.

1. What are my start positions `(x0, y0)` for player 1 and `(x1, y1)` for
   player 2?
2. What are my goal rows -- which rows count as "in the goal" for each side?
   (This repo assumes the middle three rows of a 7x5 board, i.e. `y in {1,2,3}`,
   the same for both sides.)
3. Is the grid 7 wide x 5 tall for everyone, or is the size also per-ID?

## B. The "occupy the same square" rule -- three unspecified sub-cases

A10 says: *"If the two players try to occupy the same square, only the player
with the ball will move to that square, and the other player will get the
ball,"* and *"if the two players try to swap squares, they will swap, and the
other player will get the ball."* Both clearly cover the case where the
contested square is **empty**. These three cases aren't covered:

4. **Ball-carrier blocked by a player who is *not* moving.** The carrier moves
   toward a cell the opponent is standing still in (the opponent moved into a
   wall, or otherwise stays put). My reading: the carrier is blocked (stays)
   **and loses the ball to the opponent**. Alternative: the carrier is blocked
   but **keeps** the ball, because the possession-flip only applies when the
   *opponent* is also actively moving into the contested cell. Which is intended?

5. **A non-carrier moving into a standing ball-carrier -- is that a steal?**
   The player without the ball moves into the cell the carrier is standing still
   in. My reading: the mover is blocked (stays) but **takes the ball** -- a bump
   is a steal. Alternative: the move simply fails and possession is unchanged.
   Which is intended?

6. **Possession when the "winner" can't actually move in.** Both players target
   the same cell, but that cell is occupied by the player who *doesn't* get it
   (they're standing still on it). My reading: nobody moves, but the ball still
   passes to that stationary player -- "the other player will get the ball"
   applies even though "the player with the ball" never moved. Alternative:
   since no one moved, possession is unchanged. Which is intended?

## C. Discounting and horizon

7. For solving the full Markov game (beyond Part 1's single-step reward table),
   is there an intended discount factor, or is it the undiscounted game that
   ties after 100 steps? (This repo uses gamma = 0.9 and checks 0.5-0.99; the
   pure-saddle result holds at every value tested, but the exact mixed-state
   count under Littman's random-move variant shifts a little with gamma.)

## D. Edge / clamping (likely obvious, included for completeness)

8. A non-scoring move off the board edge just leaves the player where it was
   (clamped)? And a carrier moving off its attacking edge scores only from a
   goal row -- off a non-goal row it clamps?

## Why B matters most

Questions 4-6 are the ones that can change the deterministic transition
function, and therefore Claim A ("every converged stage game has a pure
saddle"). If staff confirm the readings above, the result stands as measured; if
any differ, re-running `scripts/experiments.py baseline` with the corrected rule
in `soccer_nash/game.py` gives the updated count.
