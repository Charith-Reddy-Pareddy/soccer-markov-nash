# A second reward objective: goal rate

The default game rewards a **win**: the first goal ends the game, reward `+1`
to the scorer and `−1` to the other, and

```
V_win(s) = P(player 0 wins from s) − P(player 1 wins from s)
```

`γ < 1` is only a solver device -- the true game is undiscounted and
terminating.

`scoring="rate"` is a second objective for the *same* dynamics. A goal still
scores `+1 / −1`, but **play continues**: after a goal both players return to
their kickoff cells and **the ball goes to the team that just conceded** (as in
real soccer -- concede, and you kick off). The game runs to `max_steps`, and

```
V_rate(s) = E[ Σ_t γ^t (goal difference at step t) | start at s ]
```

Now `γ < 1` is load-bearing: it is the rate at which future goals are
discounted. `scripts/reward.py` (`make reward`) solves both.

## The restart rule is a design choice

Giving the ball to the conceding team is deliberate. It means **conceding is a
floor, not a cliff**: from a hopeless position a player can concede a goal and
restart with possession at the centre, so no state is worth less than
`−1 + γ · V_rate(restart)`. Alternatives -- ball to the scorer, or a neutral
centre bounce -- would give different strategic texture; this one keeps the game
zero-sum and self-correcting.

## What changes, and what does not

On the 7×5 board, 3-cell goal, random move order, `γ = 0.9`:

| | `scoring="win"` | `scoring="rate"` |
|---|---|---|
| no-pure-saddle (mixed) states | 94 | **94 -- the same 94** |
| value range | `[−1.00, +1.00]` | `[−0.88, +0.88]` |
| V(kickoff) | `+0.150` | `+0.132` |

**The mixed-NE region is invariant to the reward objective.** Win and rate give
the identical 94 no-pure-saddle states (0 win-only, 0 rate-only). A mixed Nash
equilibrium is an equilibrium of a *single stage game* `M(s)`; changing the
objective rescales the entries of `M(s)` through `V(s')`, but not enough to move
the `maximin = minimax` boundary. This is the concrete version of the standard
observation that mixed NEs are a property of the stage game, not of how the
horizon is scored.

**The value range compresses.** Under `win` a lost position is worth `−1`
(game over). Under `rate` the worst position is `−0.88`: the concede-and-restart
option puts a floor under every state. The whole value surface is pulled toward
`V(kickoff)` -- see `docs/figures/png/reward_value.png`.

## Deterministic dynamics

`scoring="rate"` does not create mixing on its own: the deterministic game has
0 no-pure-saddle stage games under both objectives, at every horizon
(`run_finite_horizon`). Positional determinacy survives the goal-reset. The
reward objective and the transition rule are independent axes
([design.md](design.md)).
