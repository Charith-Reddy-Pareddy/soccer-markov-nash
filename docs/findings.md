# Findings

Working notes. The polished write-up is [report.md](report.md).

## Setup

- Environment: 7x5 grid, 2380 non-terminal states. Three move-resolution rules:
  - **deterministic** (A10 rule): carrier wins contested squares, swaps flip
    possession, a carrier blocked by a standing opponent loses the ball.
    Transitions are a pure function of the joint action.
  - **random** (Littman rule): the two moves are applied in a random order, each
    ordering with probability 1/2; a move into the other player's current cell
    fails and hands over the ball if the mover was carrying it.
  - **coinflip**: the A10 rule, but a fair coin (not possession) decides who
    wins a contested square or a swap.
- Rewards: `+1 / -1 / 0`, zero-sum, immediate reward undiscounted.
- Solver: `V(s) = val(E[R(s) + gamma * V(s')])` with three stage solvers --
  `pure` (maximin), `mixed` (LP minimax), `hybrid` (pure saddle when it exists,
  LP otherwise). Run by value iteration (`run()`) or freeze-then-iterate policy
  iteration (`run_policy_iteration()`).

## Q1: Can pure-strategy Nash equilibria solve the game?

**Deterministic model: yes, for every enumerated state (Claim A).** All 2380
converged stage games have a pure saddle (`maximin == minimax`), so `pure`,
`hybrid` and `mixed` give identical value functions and value iteration
converges in 9 sweeps with no LP. Playing a pure saddle action everywhere is a
stationary pure-strategy equilibrium (Claim B). This is not a proof about the
theoretical A10 game (Claim C) -- see `docs/assumptions.md`.

Intuition: deterministic transitions + a carrier that always wins contested
squares means best responses are pure -- there is no guessing pressure.

## Q2: When does the pure equilibrium break?

**Random move order breaks it -- and the gate is the width of the goal mouth,
not the size of the board** (`experiments/board_sweep.csv`,
`goal_mouth_sweep.csv`).

| goal mouth | mixed states |
|---|---|
| 1 cell (any board 3x3 .. 11x3, or 5x9 with `goal_rows=(4,)`) | **0** |
| >= 2 cells | grows with board area |

With a one-cell goal the carrier's only winning approach is that single cell, so
the defender always knows where to stand and every stage game has a pure saddle.
With a wider goal the carrier threatens more than one cell and, under the random
move order, the defender cannot cover them all -- it has to guess. On the 7x5
board (3-cell goal), 94 of 2380 stage games have no pure saddle (3.95% *for
this configuration*).

The 94 no-saddle states come in exact mirror pairs, are within a move of
adjacency, and ~88 have the defender ahead of the carrier. Geometry predicts the
`pure` / `mixed` label with precision 0.96 / recall 0.91 -- see
[geometry.md](geometry.md).

### It is the move *order*, not the randomness, that breaks it

The **coinflip** rule is also stochastic -- a fair coin decides every contested
square and swap -- yet its Nash Q value function is *identical to the
deterministic game's* (`max |V_det - V_coin| = 0` for every gamma) and it has a
pure stationary equilibrium at every state. With optimal play neither side
enters a contest it would lose under the deterministic rule, so the coin is
never actually flipped on the equilibrium path. Only the sequential coupling of
Littman's random move order creates the matching-pennies sub-games that need
mixed strategies.

## Q3: Does using pure strategies anyway cost anything?

Yes. On the random game the `pure` (maximin) solver is a strict lower bound:

```
move_order = random, gamma = 0.9
          iters   V(kickoff)   saddle%
pure        18      +0.000       97.8%
hybrid      94      +0.150       96.0%     <- true minimax value
```

The pure-only solver **under-values the kickoff by 0.15** and sits up to `0.37`
below the true value elsewhere. Running the full LP (`mixed`) over the 7x5
random game reproduces the `hybrid` value function to `4e-16` -- the LP is only
doing work on the 94 no-saddle states (~8 min vs ~20 s).

## Q4: Does intermediate-reward shaping help?

Potential-based shaping `F = gamma*Phi(s') - Phi(s)` leaves the equilibrium
exactly where it was: `V_shaped = V_base - Phi` to machine precision, identical
optimal policies, identical `no_saddle_states`. But it *slows* exact value
iteration from 9 sweeps to ~130, because `Phi` fills in the ~1000 states where
`V_base` is exactly 0. A naive (non-potential) per-step possession bonus does
change the solution -- the kickoff draw becomes a +0.5 value for the carrier.
See [shaping.md](shaping.md).

## Q5: Is the Nash Q policy an equilibrium, and is it worth playing?

**Yes and yes** (`scripts/selfplay.py`):

- The duality gap `V0_br(s0) + V1_br(s0)` of the Nash Q policy is `< 1e-9` for
  every move order -- an optimal opponent cannot beat the equilibrium value.
- Nash vs. Nash from the kickoff reproduces the value: forced draws in the
  deterministic and coinflip games, a `+0.149 +/- 0.005` empirical discounted
  return over 5 seeds (vs. `0.150` computed) in the random game.
- The Part 2 best-response-to-the-scripted-opponent policy has exploitability
  `0.43` -- worse than moving uniformly at random (`0.39`). Best-responding to
  one assumed opponent is fragile; the Nash policy is the safe submission.

## Q6: A neural policy is only as safe as its worst state

Fitting the Nash Q policy to a bias-free `5->99->99->4` network with
partial-label training gets ~99% of states' action right, but a single state
where the network prefers a losing move lets a best-responding opponent force a
win. Trained Network First landed at exploitability ~0.3, Network Second at
~0.0 -- re-seed until the exploitability number the script prints is near 0.
See [a10_competition.md](a10_competition.md).

## Value iteration vs. policy iteration

`run_policy_iteration()` solves each stage game only on its outer rounds and
holds the strategies fixed for cheap linear evaluation sweeps in between. On the
7x5 random game it reaches the same fixed point with ~5x fewer matrix-game
solves (1879 vs 9672), but the frozen strategies stay *maximally stale* (a full
pure flip) for 16 outer rounds before locking in, so wall-clock is not lower
(17s vs 14s). Value iteration with the per-sweep Nash cache wins here.
`scripts/policy_iteration.py`.

## Q7: Numerical foundations (from the research meeting)

`scripts/numerics.py`, `docs/numerics.md`.

- **The "3 numbers" / value bracket:** with `scipy` HiGHS the row player's
  guaranteed / bilinear / best-response values agree to `1.1e-16` per stage
  game, `8.7e-10` accumulated. `value_bracket` certifies the LP error.
- **Rounding under discounting:** rounding stage-game entries to 0.1 flips the
  pure-saddle status of 62 of the random game's stage games (the small-entry
  mixed region); 0 on the deterministic game. `classify_stage_game` uses a
  scale-aware tolerance instead.
- **Degeneracy:** the deterministic game has a *non-strict* saddle at all 2380
  states; the random game has 604 strict-pure, 1682 degenerate, 94 mixed.
- **The mixed states:** 68 of the 94 reduce to a 2x2 matching-pennies support
  (carrier {advance, hold} x defender {block, intercept}).

## Q3: Does discounting change which states require mixing?

`experiments/gamma_sweep.csv`. **Yes, mildly.** On the 7x5 random game the mixed
count varies non-monotonically with `gamma`: 122 at 0.5, dips to 94 at 0.9-0.95,
back to 102 at 0.99-0.995. Deterministic stays 0 at every discount. Iterations
scale hard (26 at gamma=0.5, 373 at 0.995); the kickoff value rises monotonically
from 0.001 to 0.32 as future scoring matters more.

## RQ4: Numerical robustness

`experiments/tolerance_sweep.csv`. The `pure` / `degenerate` / `mixed` split
(604 / 1682 / 94) is **identical for every classification tolerance from 1e-12
to 1e-3** -- the boundary is well separated. It only degrades once the tolerance
approaches the actual `minimax - maximin` gaps: 80 mixed at 1e-2, 0 at 1e-1
(a 10% tolerance, equivalently rounding to one decimal, erases the whole mixed
region). The value-bracket gap is `2.2e-16` throughout.

## Value-function structure (deterministic game)

- Values fall in clean `gamma^k` bands, `k` = moves to a forced score.
- A defender parked between carrier and goal drives the value to `0` (~1000 of
  2380 states) -- a forced draw.
- Antisymmetric under role-swap + board mirror.
- `V(kick) = 0` for every discount (0.5 .. 0.99); convergence is
  discount-insensitive because values propagate along shortest paths to a
  forced score.

Reproduce: `python scripts/analyze.py --move-order {deterministic,random,coinflip}`.
