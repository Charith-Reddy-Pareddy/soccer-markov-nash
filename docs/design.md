# The game: a configurable family

This project studies *when* a two-player soccer Markov game needs mixed
strategies. To answer that, the environment cannot be a single fixed game -- it
has to be a family whose axes can be turned independently. `soccer_nash/game.py`
(`SoccerGame`) is that family. This page states every axis, the reward, and the
transition rules as **design choices of this project**, with the reason each one
is there.

Littman's 1994 soccer game and the CS 540 A10 geometry are two points in this
space, not the definition of it.

## Axes

| parameter | default | what it controls | why it is an axis |
|---|---|---|---|
| `width`, `height` | 7 × 5 | board size | tests whether mixing scales with the state space (it does not) |
| `goal_rows` | `(1, 2, 3)` | which rows score on each edge | **the** switch: a 1-cell goal is never mixed, ≥ 2 always is |
| `move_order` | `deterministic` | how simultaneous moves resolve | isolates the transition structure that creates matching pennies |
| `n_actions` | 4 | `{N,S,E,W}` or `+ stand` | Littman's fifth action; changes the *content* of a mix, not its location ([littman.md](littman.md)) |
| `scoring` | `win` | first goal ends it, or goal → reset and play on | the reward objective; the mixed region is invariant to it ([reward.md](reward.md)) |
| `p0_start`, `p1_start` | centre row, opposite ends | kickoff | the pure/mixed split is kickoff-independent; only `V(kickoff)` moves |
| `max_steps` | 100 | horizon before a tie | with the tie reward, this is the effective discount of the undiscounted game |

## Reward

Zero-sum and sparse. A carrier crossing its attacking edge on a goal row scores;
`r0 = +1` for a player-0 goal, `−1` for a player-1 goal, `0` otherwise;
`r1 = −r0`. Two objectives:

- **`scoring="win"`** (default) -- the first goal ends the game. The value is
  `P(player 0 wins) − P(player 1 wins)`; the game is undiscounted and
  terminating, and `γ < 1` in the solver is only a contraction device (the exact
  answer is 100-step backward induction, `run_finite_horizon`). The pure/mixed
  split is stable across `γ ∈ [0.5, 0.995]`.
- **`scoring="rate"`** -- a goal scores `±1` and **play continues** from a
  restart (conceding team gets the ball at the centre). The value is the
  *expected discounted goal difference*, `γ < 1` is load-bearing, and conceding
  becomes a floor rather than a cliff. The no-pure-saddle region is **identical**
  to `win`'s; the value range compresses. Full comparison in [reward.md](reward.md).

**Optional shaping** (`soccer_nash/shaping.py`) is a *training* device, kept
separate from the game's reward:

- `PotentialShaping` -- `F = γ·φ(s') − φ(s)` with `φ` = signed distance to the
  attacking goal. Potential-based, so it leaves the equilibrium and the value
  (up to the `φ` offset) exactly intact; it only speeds value iteration by
  seeding the ~1000 zero-value states.
- `StepPossessionBonus` -- a small per-step reward for holding the ball. **Not**
  potential-based: it changes the solution (`V(kickoff): 0 → +0.5`). Included as
  a cautionary example, not used in the main results.

## Transition rules

A joint action `(a0, a1)` is resolved by one of three rules. All three share the
same primitives -- clamp at walls, a carrier crossing its attacking edge on a
goal row scores -- and differ only in **who wins a contested cell and what
happens to the ball**.

### `deterministic` -- the carrier always wins

My default. Transitions are a pure function of the joint action, which makes the
game a deterministic-dynamics Markov game and lets the attractor / positional
machinery (`soccer_nash/attractor.py`) apply. Collision sub-cases:

| situation | outcome |
|---|---|
| both target the same empty cell | the carrier moves in; the other stays and **takes the ball** |
| a player targets the other's occupied cell | blocked; if the *mover* held the ball it passes to the stationary player |
| the two swap cells | they swap; possession flips to the non-carrier |
| a player targets a cell the other is vacating | both moves succeed, possession unchanged |
| out of bounds | clamped, the player stays (unless it is a score) |
| `STAND` | the player targets its own cell; the rules above do the rest |

### `random` -- Littman's rule

The two moves are applied in a uniformly random order. "A move into an occupied
cell fails, and the ball goes to the stationary player." This is the only rule
under which the game needs mixed strategies: a ball-steal now depends on the
resolution order *and* on both players' targets, which is what turns a contested
forward cell into a matching-pennies subgame.

### `coinflip` -- an ablation

The `deterministic` rule, but a fair coin (not possession) decides every
contest. This isolates *stochastic transitions* from *stochastic move order*:
`coinflip` is stochastic yet its value function is identical to `deterministic`
at every `γ`, and it keeps a pure equilibrium everywhere. Randomness alone does
not force mixing; the coupling of the outcome to both players' actions does.

## Instances

| instance | constructor | notes |
|---|---|---|
| Littman 1994 | `SoccerGame(width=5, height=4, goal_rows=(1,2), move_order="random", n_actions=5)` | γ = 0.9 ≡ his 0.1 draw probability; Figure 2 reproduced in [littman.md](littman.md) |
| CS 540 A10 geometry | `A10SoccerGame()` | 7 × 5, deterministic only, netID kickoff; the "pure saddle at every state" result is measured here. Interpreted collision sub-cases and what they would change are in [assumptions.md](assumptions.md) |
| single goal cell | `SoccerGame(goal_rows=(h//2,), move_order="random")` | the case with a partial pure-saddle proof ([proof.md](proof.md)) |

## What is a result vs. a choice

- **Choices:** the reward objective (`win` / `rate`) and its restart rule, the
  three resolution rules, the collision sub-cases, the board and goal defaults.
- **Results, robust across the choices:** a deterministic transition model has a
  pure memoryless equilibrium at every state (also with `STAND`, also under
  `rate`); the no-pure-saddle region is identical under `win` and `rate`; mixing
  under the random rule is gated perfectly by goal width and is geometrically
  local; the pure-first solver skips the LP on 96-100% of states.
