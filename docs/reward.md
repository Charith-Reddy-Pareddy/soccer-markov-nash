# Reward objectives: which ones move the mixed region

The default game rewards a **win**: the first goal ends the game, `+1` to the
scorer, `−1` to the other, and

```
V_win(s) = P(player 0 wins from s) − P(player 1 wins from s)
```

`γ < 1` is only a solver device -- the true game is undiscounted and
terminating. This page adds two more objectives for the *same* dynamics and
asks which of them leave the no-pure-saddle (mixed) region where it is.
`scripts/reward.py` (`make reward`) runs all three.

## `scoring="rate"` -- a horizon rescaling

A goal still scores `±1`, but **play continues**: both players return to their
kickoff cells and **the ball goes to the team that just conceded** (as in real
soccer). The game runs to `max_steps`, and

```
V_rate(s) = E[ Σ_t γ^t (goal difference at step t) | start at s ]
```

Now `γ < 1` is load-bearing. Giving the ball to the conceding team is a
deliberate design choice: **conceding is a floor, not a cliff** -- from a
hopeless position a player can concede and restart with possession, so no state
is worth less than `−1 + γ · V_rate(restart)`.

## `scoring="territory"` -- a dense per-step reward (this project's own)

`"win"` (first goal ends it) **plus** a per-step reward: holding the ball in the
opponent's final third earns `± territory_reward` each step, the middle third is
neutral. `territory_reward = 0.02` by default. It is a real objective -- "get
the ball into the final third and keep it there" -- not a potential-based
shaping trick, and it changes the value function *everywhere*, not just at
goals.

## Result: `rate` doesn't move the region, `territory` does

7×5 board, 3-cell goal, random move order, `γ = 0.9`:

| objective | no-pure-saddle states | shared with `win` | value range | V(kickoff) |
|---|---|---|---|---|
| `win` | 94 | -- | `[−1.00, +1.00]` | `+0.150` |
| `rate` | 94 | **94 (0 in, 0 out)** | `[−0.88, +0.88]` | `+0.132` |
| `territory` (0.05) | 112 | 73 (39 in, 21 out) | `[−1.00, +1.00]` | `+0.000` |

**`win ↔ rate`: identical region.** A mixed Nash equilibrium is an equilibrium
of a *single stage game* `M(s)`. Changing how the *horizon* is scored rescales
the entries of `M(s)` through the continuation value `V(s′)`, uniformly enough
that no stage game crosses the `maximin = minimax` boundary. This is the
concrete form of "mixed NEs are a property of the stage game, not of how the
horizon is scored."

**`territory`: the region moves.** The per-step reward is added *directly* to
`M(s)[a0, a1]` -- and it depends on where the ball ends up, which depends on
*both* players' actions. That action-coupled term can and does cross the
boundary. The drift grows with `territory_reward`:

| `territory_reward` | mixed | entered | left | V(kickoff) |
|---|---|---|---|---|
| 0.00 | 94 | 0 | 0 | `+0.150` |
| 0.01 | 106 | 14 | 2 | `+0.126` |
| 0.02 | 110 | 18 | 2 | `+0.094` |
| 0.05 | 112 | 39 | 21 | `+0.000` |
| 0.10 | 93 | 45 | 46 | `−0.131` |

## The sharper point: `territory` breaks the *deterministic* game

Under `win` and `rate` the deterministic game is positionally determined -- a
pure memoryless equilibrium at every state, 0 no-pure-saddle stage games
([proof.md](proof.md)). Under `territory` (0.05) the deterministic game has
**69 genuinely mixed stage games** (gap median `0.070` -- *deeper* than the
random game's, entropy median `0.78` bits, all 69 with a 2×2 matching-pennies
core), and `V(kickoff) = −0.16`.

So `territory` does to the deterministic game what Littman's random move order
does to the goal-width game, by a different route. The move order couples the
*transition* outcome to both actions; the territory reward couples the
*immediate reward* to both actions. Either coupling can turn a stage game into
matching pennies. `slip` and `tackle` ([generalize.md](generalize.md),
[tackle.md](tackle.md)) are the transition-side versions; `territory` is the
reward-side version.

## What the mixed region *is* invariant to

| change | effect on the region | why |
|---|---|---|
| `win` → `rate` (horizon scoring) | **none** | rescales `M(s)` through `V(s′)`, uniformly |
| kickoff position | **none** | changes `V(kickoff)` only, not any `M(s)` |
| discount `γ` | a few percent | shifts the `γ^k` bands slightly |
| `territory` (dense per-step reward) | **moves it** | adds an action-coupled term to `M(s)` |
| move-resolution rule | **moves it** | adds an action-coupled term to `M(s)` via the transition |

The pattern: the mixed region is a function of the *ordinal* structure of each
stage game `M(s)`. Anything that only rescales `M(s)` through the continuation
value leaves it alone; anything that adds a term depending on *both* actions can
move it.
