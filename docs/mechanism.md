# Why a one-cell goal forces pure strategies -- an argument

The board / goal-mouth sweeps establish empirically:

- **Single-cell goal:** 0 stage games need mixed strategies, across every board
  from 3x3 to 11x5 and every discount from 0.5 to 0.99
  (`experiments/one_cell_goal.csv`, 19/19 configurations).
- **Two-or-more-cell goal (with board room):** always some mixed states, count
  growing with area (`board_sweep.csv`, `goal_mouth_sweep.csv`).

This page sketches *why*. It is an argument, not a formal proof -- Claim C in
`docs/assumptions.md` stays open.

## Setup

Under the random move order, a joint action `(a0, a1)` from state `s` yields a
lottery over the two resolution orders, each with probability 1/2. The stage
matrix is

    M(s)[a0, a1] = E_order[ r(s, a0, a1) + gamma * V(s') ].

A stage game lacks a pure saddle iff the best-response correspondence *cycles*:
player 0's best row changes with player 1's column and player 1's best column
changes with player 0's row (a 2x2 matching-pennies structure -- confirmed for
68 of the 94 mixed states).

## The single-cell-goal argument

Let the goal be the single cell `T = (edge, g)`. To win, the carrier must occupy
`T` with the ball.

1. **One threat.** The carrier has exactly one winning target, `T`. Its distance
   to a win is `d_c = dist(carrier, T)` (Manhattan, since only the carrier can
   pass through `T`). The defender's job is to keep `d_c` from reaching 0 faster
   than it can interpose; its relevant quantity is `d_d = dist(defender, T)`.

2. **The value is a race.** With one target, `V(s)` is monotone in `d_c - d_d`
   and possession: the carrier is winning iff it can reach `T` strictly before
   the defender can block the last step. Every reachable "position" collapses to
   this one-dimensional race plus who holds the ball.

3. **Directions do not depend on the coin.** In the race, the carrier's
   value-maximising move always *decreases* `d_c`; the defender's always
   *decreases* `d_d` (or holds `T`). The random move order changes *timing* --
   who resolves first on a contested square -- but not *which* cell each player
   wants, because there is only one cell worth contesting. So each player has a
   direction that is a best response regardless of the other's action: a pure
   saddle.

4. **Contested squares are value-neutral on the equilibrium path.** A contest
   only occurs on the single path to `T`. The carrier gains nothing by
   contesting a square it would lose (it stays behind in the race), and the
   defender gains nothing by contesting one it would lose; equilibrium play
   avoids the coin, exactly as in the deterministic and coin-flip games.

## Why >= 2 cells breaks it

With goal cells `T_1, T_2` (adjacent, same edge), the carrier near the goal can
threaten *both*. The defender, one interception-move away, can cover `T_1` or
`T_2` but not both. Now the carrier's best "which cell" choice depends on the
defender's cover and vice versa -- the cyclic structure. The geometry model
picks this up precisely: `defender_can_intercept` predicts `mixed` at precision
0.96, and it can only be true when there is more than one threatened cell for
the defender to be *between*.

## Progress on a proof

[proof.md](proof.md) turns this sketch into:

- **the defender's closed-form optimal strategy** (guard the one goal cell),
  verified to secure `V*` from every state -- which proves `minimax(M_s) = V*(s)`
  by weak duality;
- **a dominance-solvability certificate**: every single-cell stage game reduces
  to a pure saddle by iterated weak-dominance elimination
  (`soccer_nash/dominance.py`), machine-checked for all boards up to 11x5 and all
  discounts 0.5-0.99.

Together these prove the theorem for every finite board. The remaining gap is a
board-size-free argument for the carrier's half (`maximin(M_s) = V*(s)` via a
closed-form carrier strategy, or a termination proof for the elimination).
