# Pure vs. Mixed Nash Q-Iteration for the Soccer Markov Game

## Research question

> Can pure-strategy Nash equilibria be used to *efficiently* solve the soccer
> Markov game, and under what state and reward settings does a pure Nash
> equilibrium exist?

Short answers (see `docs/assumptions.md` for the exact scope of each claim):

1. **For the implemented deterministic A10 transition model, every one of the
   2380 enumerated non-terminal stage games -- built from the converged value
   function -- has a pure saddle point.** Playing a pure saddle action at every
   state is then a stationary pure-strategy equilibrium (Claim B), found by
   value iteration in 9 sweeps with no linear program. This does *not* prove the
   theoretical A10 game necessarily has one (Claim C) -- that depends on
   confirming the interpreted collision sub-cases and holds only for the reward
   settings tested.
2. **Randomness in the *outcome* of a contested square (a coin-flip tie-break)
   does not change this.** Its value function is bit-identical to the
   deterministic game's.
3. **Littman's random *move order* does break it**: under the specified 7x5
   configuration, 94 of 2380 stage games (3.9%) have no pure saddle, so no
   stationary pure-strategy equilibrium exists and a mixed (LP) solver is
   required for those states. The 3.9% is not a universal constant -- it is
   board-size dependent, and small or narrow boards have none.
4. **Potential-based reward shaping preserves the equilibrium structure
   exactly**: the value function shifts by `-Phi`, but the pure/mixed
   classification of every state is unchanged. A naive per-step possession bonus
   changes the solution.
5. The recommended solver is the **hybrid**: take the pure saddle value wherever
   `maximin == minimax`, fall back to the LP only where it does not. It matches
   the full LP solution to `4e-16` while doing ~25x less work. (A fourth backup,
   `pure`, takes the maximin security value even with no saddle -- a fast lower
   bound, not a Nash solver.)

---

## 1. The game

A two-player zero-sum Markov game on a 7x5 grid (CS 540 A10 geometry;
`soccer_nash/game.py`).

- **State** `(x0, y0, x1, y1, b)`: the two players' cells and the ball carrier
  `b in {0, 1}`. Terminal states are `(-1, -1, -1, -1, winner)`. 2380
  non-terminal states.
- **Actions**: `U, D, L, R`, chosen simultaneously.
- **Reward**: `+1` win / `-1` loss / `0` otherwise, zero-sum, immediate reward
  undiscounted. Discount `gamma < 1` is used for training (as A10 suggests).
- **Move resolution** -- three rules:
  - `deterministic` (A10): the carrier wins a contested square, a swap flips
    possession, a carrier blocked by a standing opponent loses the ball.
  - `random` (Littman): the two moves are applied in a random order (1/2 each);
    a move into the other player's current cell fails and transfers the ball.
  - `coinflip`: the A10 rule, but a fair coin -- not possession -- decides who
    wins a contested square or a swap.

## 2. The two solvers

The game is zero-sum, so at every state the stage game is a matrix
`M(s)[a0, a1] = E[ R + gamma * V(s') ]` and the Shapley (1953) fixed point

```
V(s) = val(M(s)) = max_{p in Delta} min_{a1} (p^T M(s))[a1]
                 = min_{q in Delta} max_{a0} (M(s) q)[a0]
```

is the minimax value function. Value iteration applies this operator; the
`Q = R + beta * Nash(Q')` update from the project notes is the same thing.

`soccer_nash/matrix_games.py` provides both stage solvers:

| version | stage-game value | how |
|---|---|---|
| **Pure Nash Q-iteration** (`mode="pure"`) | pure maximin `max_i min_j M[i,j]` | `security_strategy_row` -- O(A^2), no LP |
| **LP / Mixed Nash Q-iteration** (`mode="mixed"`) | LP minimax value | `game_value` via `scipy.optimize.linprog` (HiGHS) |
| **Hybrid** (`mode="hybrid"`) | pure saddle value if `maximin == minimax`, else LP | the recommended default |

A pure saddle point exists iff `max_i min_j M[i,j] == min_j max_i M[i,j]`
(`pure_bounds`). When it does, the pure value *is* the minimax value, so the
hybrid solver is exact; when it does not, the hybrid falls back to the LP.

Both `run()` (value iteration) and `run_policy_iteration()` (freeze the stage
strategies, run cheap linear evaluation sweeps, re-solve -- the notes' "do not
recompute Nash every time") are implemented and report their matrix-game solve
counts.

## 3. When does a pure Nash equilibrium exist?

### 3.1 By move rule (7x5 board, gamma = 0.9)

| move order | stochastic? | converged stage games with no pure saddle | stationary pure eq? | `V(kickoff)` |
|---|---|---|---|---|
| `deterministic` | no | 0 / 2380 | yes (Claim B) | 0 |
| `coinflip` | yes | 0 / 2380 | yes (Claim B) | 0 |
| `random` | yes | 94 / 2380 (3.9% here) | no | +0.150 |

The deterministic and coinflip value functions are *identical*
(`max |V_det - V_coin| = 0` for every gamma tested). With optimal play neither
player enters a contest it would lose under the deterministic rule, so the coin
is never flipped on the equilibrium path.

It is therefore not stochasticity that forces mixed strategies -- it is the
*sequential coupling* of Littman's random move order, where whether you steal
the ball depends on who is resolved first *and* on both players' targets. That
creates matching-pennies sub-games with no pure saddle.

### 3.2 By board size (random move order)

| board | states | no-saddle states | pure equilibrium? |
|---|---|---|---|
| 4x3, 1 goal row | 264 | 0 | exists |
| 5x3, 1 goal row | 420 | 0 | exists |
| 5x5, 3 goal rows | 1200 | 54 (4.5%) | fails |
| 7x5, 3 goal rows | 2380 | 94 (3.9%) | fails |

The no-saddle states on the 7x5 board come in exact mirror pairs (47 with each
ball holder), have the players adjacent or one step from it, and concentrate on
the "defender directly ahead of the carrier, same row" stand-off. They never
occur when the players are far apart. Narrow boards do not have room for that
geometry, so their equilibrium stays pure.

### 3.3 By reward setting

Potential-based shaping `F(s, s') = gamma * Phi(s') - Phi(s)` with
`Phi` = possession bonus + ball advancement:

- `V_shaped = V_base - Phi` to machine precision; identical policies; identical
  `no_saddle_states`. **The equilibrium -- pure or mixed -- is unchanged.**
- It *slows* exact value iteration (9 -> ~130 sweeps) because `Phi` is nonzero
  on the ~1000 states where `V_base = 0`.

A non-potential per-step possession bonus is *not* invariant: a `+0.05` bonus
turns the kickoff draw into a `+0.5` value for the carrier and changes the
optimal policy. Shaping rewards for this game must be potential-based.

## 4. Can pure-strategy Nash equilibria solve it efficiently?

**Yes.** On the deterministic and coinflip games the pure solver *is* the exact
solver: every stage game has a pure saddle, so `pure`, `hybrid` and `mixed`
return the same value function and value iteration converges in 9 sweeps with
zero LP calls.

On the random game the hybrid solver is still the right tool:

```
random move order, gamma = 0.9
          sweeps   V(kickoff)   matrix-game solves   wall clock
pure         18      +0.000            0                 <1 s      (wrong: a lower bound)
hybrid       94      +0.150         ~8 800               ~20 s     (exact)
mixed        94      +0.150        ~3.6 M                ~8 min    (exact, wasteful)
```

- `hybrid` reproduces the full-LP (`mixed`) value function to `4e-16` while
  solving an LP only on the ~4% of states that need one.
- The `pure`-only solver is a strict lower bound: it reports a kickoff draw when
  the game actually favours the initial carrier by `0.15`, and sits up to `0.37`
  below the true value elsewhere. Pure-strategy iteration is efficient but, when
  a mixed equilibrium is the real answer, wrong.

### Freeze-then-iterate, and its staleness

`run_policy_iteration()` solves each stage game only on its outer rounds,
holding the strategies fixed for cheap linear evaluation sweeps in between. It
reaches the same fixed point (`|diff| < 3e-9`) with ~5x fewer matrix-game solves
(1 879 vs 9 672 on the random game) -- but it records the *staleness* the
scheme's proposer flagged: how far the frozen strategies are from the Nash of
the current Q. That stays **maximal (a full pure-strategy flip) for 16 outer
rounds**, then snaps to `< 1e-8`. The thrashing eats the wall-clock saving
(17 s vs 14 s), and on the deterministic game freeze-then-iterate is strictly
worse (65 LP solves where value iteration needs 0). Value iteration with the
per-sweep Nash cache -- which `run()` already does -- wins here; the freeze
trick pays off only where the equilibrium solve dominates the backup.

## 4b. Numerical foundations

Three things the research meeting flagged as unsolved for the discrete solver
(`soccer_nash/numerics.py`, `scripts/numerics.py`, `docs/numerics.md`):

- **The value is three numbers** -- what the row player guarantees, gets, and
  can reach. `value_bracket` returns all three; the true minimax value is
  bracketed, so `upper - lower` certifies the LP error. On the random game the
  gap is `1.1e-16` per stage and `8.7e-10` accumulated: with `scipy`'s HiGHS the
  concern is real in theory, negligible in practice.
- **Rounding to force indifference is scale-blind.** `classify_stage_game`
  splits stage games into `pure` (strict saddle), `mixed`, and `degenerate`
  (non-strict saddle) with a tolerance that scales with the entries.
  `rounding_changes_saddle` shows fixed 0.1-rounding flips the saddle status of
  **62** of the random game's stage games -- exactly the small-entry mixed
  region -- while the deterministic game (clean `gamma^k` bands) is safe to
  round but has a **non-strict saddle at every one of its 2 380 states**.
- **The mixed states have a 2x2 matching-pennies support.** Of the 94 states
  that need mixing, **68 have an equilibrium supported on a 2x2 subgame** with no
  pure saddle (matching-pennies strategic structure): carrier {advance, hold} vs
  defender {block, intercept}. Iterated *strict* dominance only collapses 4 of
  them fully -- the rest carry tied actions -- so this is a statement about the
  equilibrium support, not a reduction of the full game. All 94 have the same
  value across every equilibrium (zero-sum interchangeability), and 64 have a
  unique equilibrium.

## 5. The equilibrium policy is worth playing

Self-play and exact best-response analysis (`soccer_nash/exploit.py`,
`scripts/selfplay.py`):

- The Nash Q policy has a duality gap `V0_br(s0) + V1_br(s0) < 1e-9` for every
  move order -- **no opponent can beat the game value.**
- Nash vs. Nash reproduces the value: forced draws in the deterministic and
  coinflip games; a `+0.157` empirical discounted return (vs. `+0.150`
  computed) in the random game.
- Best-responding to a *guess* about the opponent is fragile: the Part 2
  best-response-to-the-scripted-opponent policy has exploitability `0.43` --
  worse than moving uniformly at random (`0.39`) once the opponent
  best-responds to it. This is exactly the A10 competition's warning about
  non-equilibrium submissions.

## 6. A10 deliverables

- **Part 1** (`scripts/a10_part1.py`): successor states (Q1/Q3) and transition
  rewards (Q2/Q4) for any state, in the A10 action order.
- **Part 2** (`scripts/a10_part2.py`): the exact best response to the scripted
  opponent, imitated by a bias-free `5->99->99->4` ReLU/softmax network
  (partial-label training, ~99% optimal actions), plus a self-consistent 7-step
  winning Q8 trajectory.
- **Competition** (`scripts/a10_competition.py`): the two equilibrium networks
  (player 0 and player 1) fitted to the Nash Q solution. The exact Nash policy
  is unexploitable; the network approximation is only as safe as its worst-fit
  state (trained Network First reached exploitability ~0.28), so the script
  prints that number and the student re-seeds until it is near 0.

## 7. Conclusion

For the soccer game as specified by A10, pure-strategy Nash equilibria are not
just usable but *sufficient and efficient*: they exist everywhere and are found
in a handful of sweeps with no linear programming. Pure equilibria fail only
under Littman's sequential random move order, and even then only on ~4% of
states of a large enough board. The hybrid solver -- pure where possible, LP
where necessary -- is exact everywhere and pays the LP cost only where a pure
equilibrium genuinely does not exist. Reward shaping, if potential-based, leaves
all of this untouched.
