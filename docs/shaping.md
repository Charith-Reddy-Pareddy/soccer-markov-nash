# Intermediate-reward shaping

The A10 write-up suggests adding "small positive rewards for possessing the ball
and moving closer to the goal" on top of the sparse `+1 / -1` win/lose signal.
`soccer_nash.shaping` implements two versions and both solvers
(`NashQIteration`, `BestResponse`) take a `shaping=` argument.

## `PotentialShaping` -- potential-based, solution-preserving

Adds `F(s, s') = gamma * Phi(s') - Phi(s)` to player 0's reward (and its
negative to player 1's), with

```
Phi(s) = w_ball * (+1 if player 0 has the ball else -1)
       + w_advance * (signed ball advancement toward player 0's goal)
```

Potential-based shaping (Ng, Harada & Russell 1999; Devlin & Kudenko 2011 for
Markov games) provably preserves every Nash equilibrium. Verified numerically:

| solver | `max |V_shaped(s) - (V_base(s) - Phi(s))|` | policy / equilibrium |
|---|---|---|
| `BestResponse` | `2e-16` | identical optimal actions, still a 7-step win |
| `NashQIteration` (hybrid) | `9e-10` | identical `no_saddle_states` set |

At convergence the shaped stage game is exactly `M_base - Phi(s)` -- a constant
shift per state -- so saddle points, best responses and mixed equilibria are
untouched and only the value function moves, by `-Phi`.

### ...but it slows exact value iteration

`Phi` is nonzero almost everywhere, whereas `V_base` is exactly `0` on ~1000 of
the 2380 states (a defender neutralises the carrier). Shaping destroys that
sparsity, so plain value iteration has to run its full contraction horizon:

```
gamma = 0.9, hybrid, tol 1e-8
  no shaping            9 sweeps
  w_ball only         133 sweeps
  w_advance only      131 sweeps
  both (0.02 each)    123 sweeps
```

Potential-based shaping helps *sample-based* RL by guiding exploration; for
dynamic programming on this game it is pure overhead. It earns its place here
only as a correctness check on the equilibrium machinery.

## `StepPossessionBonus` -- naive, solution-changing

Adds a flat `+/- bonus` every step player 0 holds the ball, with no potential
structure. This is *not* invariant: on the 5x3 board the kickoff value jumps
from `0.00` (a forced draw) to `0.50` -- the carrier is now paid to keep the
ball instead of having to score, and the optimal policy changes accordingly. A
cautionary example of why shaping rewards should be potential-based.
