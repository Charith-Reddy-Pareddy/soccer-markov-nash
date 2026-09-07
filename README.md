# Soccer Markov Game: Pure vs. Mixed Nash Q-Iteration

Research code built on top of the CS 540 A10 soccer assignment
(<https://pages.cs.wisc.edu/~yw/CS540S26A10.html>).

## Research question

Can **pure-strategy Nash equilibria** be used to solve the soccer Markov game
efficiently, and under which state / reward settings does a pure equilibrium of
the stage game exist? When it does not, we fall back to a mixed-strategy
(linear-programming) solver.

## Plan

1. **Environment** — reproduce the A10 two-player soccer Markov game on a 7x5
   grid. Two move-resolution rules: `deterministic` (A10) and `random`
   (Littman's random move order).
2. **Stage-game solvers** — pure saddle-point finder and an LP solver for
   zero-sum mixed Nash.
3. **Nash Q-iteration** — three modes:
   - `pure` — pure security value at every stage game.
   - `mixed` — exact LP minimax value at every stage game.
   - `hybrid` — pure equilibrium when it exists, LP otherwise.
4. **Analysis** — where the value functions and policies agree / diverge.

## A10 Part 1

`python scripts/a10_part1.py x0,y0,x1,y1,b` prints the 16 successor states
(Q1/Q3) and the 16 transition rewards (Q2/Q4). See
[docs/a10_part1.md](docs/a10_part1.md).

## A10 Part 2

`python scripts/a10_part2.py --out results/` solves the exact best response to
the scripted opponent, imitates it with a bias-free `5->99->99->4` network, and
writes the Q7 weights and the Q8 winning trajectory. See
[docs/a10_part2.md](docs/a10_part2.md).

## A10 competition

`python scripts/a10_competition.py --out results/` fits the two equilibrium
policy networks (player 0 and player 1) to the Nash Q solution and reports how
exploitable the trained networks are. The Nash policy is unbeatable below the
game value; the network approximation is only as safe as its worst-fit state.
See [docs/a10_competition.md](docs/a10_competition.md).

## Self-play

`python scripts/selfplay.py` plays the Nash Q policy against itself (empirical
return matches the value function) and reports exploitability: the Nash policy
has a `~0` duality gap, while best-responding to one assumed opponent is more
exploitable than random play. See [docs/selfplay.md](docs/selfplay.md).

## Reward shaping

`soccer_nash.shaping` adds intermediate rewards on top of the sparse win/lose
signal. Potential-based shaping is verified to leave the equilibrium unchanged
(value shifts by exactly `-Phi`); a naive per-step possession bonus is shown to
change the solution. See [docs/shaping.md](docs/shaping.md).

## Headline result

- **Deterministic game:** every stage game has a pure saddle; `pure`, `hybrid`
  and `mixed` agree exactly. A pure stationary Nash equilibrium exists.
- **Random move order:** ~4% of stage games (7x5 board) have no pure saddle, so
  no pure stationary equilibrium exists; the `pure` solver then under-values the
  kickoff by 0.15. Small/narrow boards keep a pure equilibrium.
- **Coinflip tie-break:** stochastic, yet its value function is identical to the
  deterministic game's and it keeps a pure equilibrium everywhere -- it is
  Littman's move *order*, not randomness, that forces mixed strategies.

See [docs/findings.md](docs/findings.md).

## Environment assumptions

The A10 page parameterises start positions and goal rows by student ID (redacted
in the public page). This repo uses the standard Littman soccer geometry:

- Grid: width 7 (`x` in `0..6`), height 5 (`y` in `0..4`).
- Player 0 starts left and scores through the right edge (`x = 7`); player 1
  starts right and scores through the left edge (`x = -1`).
- Goal mouth: middle rows `y in {1, 2, 3}`.
- Actions: `U, D, L, R` (no stay). Vertical moves clamp at the top/bottom walls;
  horizontal moves clamp except a ball carrier moving into the opponent goal.
- Collisions (`deterministic`): if both players target the same square, the ball
  carrier takes it and possession flips to the other player; on a swap, players
  swap and possession flips. A carrier blocked by a stationary opponent stays
  put and loses the ball.
- Collisions (`random`): the two moves are applied in a random order (each with
  probability 1/2); a move into the other player's current cell fails, and
  transfers the ball if the mover held it.
- Collisions (`coinflip`): the A10 rule, but a fair coin (not possession)
  decides who wins a contested square or a swap.
- Rewards: `+1` win, `-1` loss, `0` otherwise (zero-sum). Game ties after 100
  steps.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

On very recent macOS builds the PyPI SciPy wheel can fail to load
(`_spropack.so` dyld error). If so, create the venv against a working
interpreter instead:

```bash
python -m venv --system-site-packages .venv
```
