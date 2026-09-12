# Littman (1994) and the fifth action

Littman's *Markov games as a framework for multi-agent reinforcement learning*
introduced the soccer game this project studies. His formulation differs from
the one used elsewhere in this repo in two ways that matter for mixed
strategies:

| | Littman 1994 | this repo's default |
|---|---|---|
| grid | 4 rows × 5 columns | configurable (7 × 5) |
| actions | **N, S, E, W, stand** (5) | N, S, E, W (4) |
| move resolution | random order, "move into an occupied cell fails and the ball goes to the stationary player" | same (`move_order="random"`) |
| draw probability | 0.1 per step | `γ = 0.9` (equivalent) |

His **Figure 2** is the canonical example that "soccer needs mixed strategies":
the ball carrier, pinned against its own goal by an adjacent defender, has a
*mixed* optimal policy -- it must randomize between **standing still** and
moving away, because any deterministic choice is exploited by the defender.

`scripts/littman.py` (`make littman`) reproduces this. It solves Littman's board
(`SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random")`) exactly
with the hybrid Nash-Q solver, with and without the stand action.

## What the stand action changes

```
 actions       states  mixed  stand in support  V(kickoff)
 N,S,E,W          760     56                 0      +0.147
 N,S,E,W,stand    760     56                48      +0.172
```

- **Not the location of mixing.** 56 stage games lack a pure saddle either way,
  and 48 of the 56 no-pure-saddle states are the same states (8 swap in, 8 swap
  out at the margin). Mixing is forced by the same geometry -- a wider-than-one
  goal, a stochastic resolution order, a defender who can contest one lane but
  not both -- and STAND is not part of that condition.
- **The content of the mix.** With five actions, the carrier's equilibrium puts
  positive weight on `STAND` in 48 of the 56 mixed states. Standing is a
  strictly better hedge than committing to a move action when the defender is
  guessing which lane you will take.
- **The value.** Standing is a real option, so every player is a little better
  off: `V(kickoff)` rises `+0.147 → +0.172`.

## The Figure 2 state

```
state (0, 1, 1, 1, 1)   player 1 carries, V = -0.315
  carrier support:  U 50%,  STAND 50%
  defender support: U 50%,  STAND 50%
  stage game (rows = carrier's U / STAND, cols = defender's U / STAND;
              carrier maximises):
              U        STAND
      U     +0.284    +0.347
      STAND +0.347    +0.284
```

The carrier is at its own goal edge with the defender one cell away. The 2 × 2
stage game is a clean **matching-pennies** matrix: the off-diagonal (players
mismatch) pays the carrier `+0.347`, the diagonal (defender guesses right)
`+0.284`, so there is no pure saddle and the unique equilibrium is
`(½ U, ½ STAND)` for both players. This is Littman's Figure 2,
drawn in `docs/gallery.html` (`littman_fig2.svg`) and contrasted with the
four-move equilibrium at the same state (`littman_stand.svg`).

## Where this leaves the main result

The four-action deterministic game still has a pure saddle at every state, and
so does the **five-action** deterministic game (`test_stand.py`) -- STAND does
not break positional determinacy. The stand action matters only for the random
move order, and there it deepens rather than moves the mixed region.
