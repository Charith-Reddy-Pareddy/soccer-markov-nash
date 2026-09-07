# Findings

## Setup

- Environment: 7x5 grid, deterministic simultaneous-move resolution as specified
  by A10 (carrier takes contested squares, swaps flip possession, a carrier
  blocked by a standing opponent loses the ball).
- Rewards: `+1 / -1 / 0`, zero-sum. Discount `gamma` swept in `[0.5, 0.99]`.
- Solver: value iteration on the stage-game value operator
  `V(s) = val(R(s) + gamma * V(succ(s)))`, with three stage solvers
  (`pure` maximin, `mixed` LP minimax, `hybrid` pure-then-LP).

## Q1: Does a pure-strategy Nash equilibrium solve this game?

**Yes, for the deterministic A10 formulation.** At every one of the 2380
non-terminal states, the stage game induced by the converged value function has
a pure saddle point (`saddle_fraction = 1.00` for every gamma tested). The
`hybrid` and `mixed` value functions are identical to `1e-9`. So the LP solver
is never actually needed here: a pure stationary equilibrium exists and value
iteration finds it in under 15 sweeps.

Intuition: with deterministic transitions and a carrier that always wins
contested squares, best responses are pure — there is no "guessing" pressure
that would force randomisation.

## Q2: When does the pure equilibrium break?

Expected to break once the transition is stochastic (e.g. Littman's random
move-order resolution) or once simultaneous-move ties are resolved by a coin
flip. Those variants are the next step; the `mixed` / `hybrid` split in the
solver is built for exactly that case.

## Value-function structure

- Values fall in clean `gamma^k` bands with `k` = moves-to-forced-score.
- A defender parked between the carrier and the goal drives the value to `0`
  (~1000 of 2380 states), i.e. a forced draw.
- The value function is antisymmetric under role-swap + board mirror.

## Discount sweep

From `python scripts/analyze.py`:

```
 gamma  iters   saddle%   V(kick)  |hyb-mix|
--------------------------------------------
  0.70      9   100.00%    0.0000   0.00e+00
  0.90      9   100.00%    0.0000   0.00e+00
  0.99      9   100.00%    0.0000   1.11e-16
```

`V(kick) = 0` at the symmetric kickoff for every discount: with best play the
deterministic game is a draw from the centre. Convergence is discount-insensitive
(9 sweeps throughout) because values propagate along shortest paths to a forced
score.
