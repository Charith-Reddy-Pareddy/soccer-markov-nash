# Findings

## Setup

- Environment: 7x5 grid. Three move-resolution rules:
  - **deterministic** (A10 rule): carrier wins contested squares, swaps flip
    possession, a carrier blocked by a standing opponent loses the ball.
    Transitions are a pure function of the joint action.
  - **random** (Littman rule): the two moves are applied in a random order, each
    ordering with probability 1/2; a move into the other player's current cell
    fails and hands over the ball if the mover was carrying it.
  - **coinflip**: the A10 rule, but a fair coin (not possession) decides who
    wins a contested square or a swap.
- Rewards: `+1 / -1 / 0`, zero-sum, immediate reward undiscounted.
- Solver: value iteration on `V(s) = val(E[R(s) + gamma * V(s')])` with three
  stage solvers -- `pure` (maximin), `mixed` (LP minimax), `hybrid` (pure saddle
  when it exists, LP otherwise). `hybrid` returns the exact minimax value by
  construction.

## Q1: Can pure-strategy Nash equilibria solve the game?

**Deterministic game: yes, completely.** At all 2380 non-terminal states the
stage game induced by the converged value function has a pure saddle point
(`saddle_fraction = 1.00`). `pure`, `hybrid` and `mixed` produce identical value
functions (max difference `0`). A pure stationary equilibrium exists and value
iteration finds it in 9 sweeps. The LP solver is never needed.

Intuition: deterministic transitions + a carrier that always wins contested
squares means best responses are pure -- there is no guessing pressure.

## Q2: When does the pure equilibrium break?

**Random move order breaks it, but only partially and only on large enough
boards.**

| board | states | no-saddle states | pure eq? |
|---|---|---|---|
| 4x3, 1 goal row | 264 | 0 | exists |
| 5x3, 1 goal row | 420 | 0 | exists |
| 5x5, 3 goal rows | 1200 | 54 (4.5%) | **fails** |
| 7x5, 3 goal rows | 2380 | 94 (3.9%) | **fails** |

On the full 7x5 random game, 94 of 2380 stage games have no pure saddle, so **no
pure stationary equilibrium exists** -- the mixed/LP solver is required. The
no-saddle states:

- come in exact mirror pairs (47 with each ball holder);
- 40 have the players adjacent, 54 one move from adjacency;
- 36 are the "defender directly ahead of the carrier, same row" stand-off -- the
  contested-square situation where the random order makes the outcome a coin
  flip and both sides want to randomise;
- never occur when the players are far apart (values there are unaffected by the
  move-order coin flip).

Small or narrow boards have no such states: the contested-square geometry that
forces mixing needs room to arise.

### It is the move *order*, not the randomness, that breaks it

The **coinflip** rule is also stochastic -- a fair coin decides every contested
square and swap -- yet its Nash Q value function is *identical to the
deterministic game's* (`max |V_det - V_coin| = 0` for every gamma) and it has a
pure stationary equilibrium at every state. With optimal play neither side
enters a contest it would lose under the deterministic rule, so the coin is
never actually flipped on the equilibrium path.

So randomness in the *outcome* of a contest changes nothing; only the
sequential coupling of Littman's random move order -- where whether you steal
the ball depends on who is resolved first *and* on both targets -- creates the
matching-pennies sub-games that need mixed strategies.

## Q3: Does using pure strategies anyway cost anything?

Yes. On the random game the `pure` (maximin) solver is a strict lower bound and
the bound is loose exactly where it matters:

```
move_order = random, gamma = 0.9
          iters   V(kickoff)   saddle%
pure        18      +0.000       97.8%
hybrid      94      +0.150       96.0%     <- true minimax value
```

The pure-only solver **under-values the kickoff by 0.15** (it reports a draw; the
game actually favours the initial carrier) because it ignores the value the
carrier gains by randomising through contested cells. Across all states the
`pure` value function sits up to `0.37` below the true value.

Running the full LP (`mixed`) over the 7x5 random game reproduces the `hybrid`
value function to `4e-16` -- confirmation that `hybrid` is exact and that the LP
is only doing work on the 94 no-saddle states (it takes ~8 min vs ~20 s).

## Q5: Is the Nash Q policy actually an equilibrium, and is it worth playing?

**Yes and yes.** Self-play (`scripts/selfplay.py`):

- The duality gap `V0_br(s0) + V1_br(s0)` of the Nash Q policy is `< 1e-9` for
  every move order -- an optimal opponent cannot beat the equilibrium value.
- Nash vs. Nash from the kickoff reproduces the value: forced draws in the
  deterministic and coinflip games, a `+0.157` empirical discounted return
  (vs. `0.150` computed) in the random game.
- The Part 2 best-response-to-the-scripted-opponent policy has exploitability
  `0.43` -- worse than moving uniformly at random (`0.39`) once the opponent
  best-responds to it. Best-responding to one assumed opponent is fragile; the
  Nash policy is the safe submission.

## Q6: A neural policy is only as safe as its worst state

The competition submission is the Nash Q policy (exploitability 0). Fitting it to
a bias-free `5->99->99->4` network with partial-label training gets ~99% of
states' action right, but the residual matters: a single state where the network
prefers a losing move lets a best-responding opponent force a win. Trained
Network First landed at exploitability ~0.3, Network Second at ~0.0 -- re-seed
and retrain until the exploitability number the script prints is near 0.
See `docs/a10_competition.md`.

## Value-function structure (deterministic game)

- Values fall in clean `gamma^k` bands, `k` = moves to a forced score.
- A defender parked between carrier and goal drives the value to `0` (~1000 of
  2380 states) -- a forced draw.
- Antisymmetric under role-swap + board mirror.

## Discount sweep (deterministic)

```
 gamma  iters   saddle%   V(kick)  |hyb-mix|
  0.70      9   100.00%    0.0000   0.00e+00
  0.90      9   100.00%    0.0000   0.00e+00
  0.99      9   100.00%    0.0000   1.11e-16
```

`V(kick) = 0` for every discount: with best play the deterministic game is a
draw from the centre. Convergence is discount-insensitive because values
propagate along shortest paths to a forced score.

Reproduce: `python scripts/analyze.py --move-order {deterministic,random}`.

## Q4: Does intermediate-reward shaping help?

Potential-based shaping `F = gamma*Phi(s') - Phi(s)` leaves the equilibrium
exactly where it was: `V_shaped = V_base - Phi` to machine precision, identical
optimal policies, identical `no_saddle_states`. But it *slows* exact value
iteration on this game from 9 sweeps to ~130, because `Phi` fills in the ~1000
states where `V_base` is exactly 0. A naive (non-potential) per-step possession
bonus does change the solution -- the kickoff draw becomes a +0.5 value for the
carrier. See `docs/shaping.md`.
