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
   grid with deterministic simultaneous-move dynamics.
2. **Stage-game solvers** — pure saddle-point finder and an LP solver for
   zero-sum mixed Nash.
3. **Nash Q-iteration** — two versions:
   - *Pure Nash Q-iteration* — pure security value at every stage game.
   - *LP / Mixed Nash Q-iteration* — exact minimax value at every stage game.
   - *Hybrid* — pure equilibrium when it exists, LP otherwise.
4. **Analysis** — where the two value functions and policies agree / diverge.

## Environment assumptions

The A10 page parameterises start positions and goal rows by student ID (redacted
in the public page). This repo uses the standard Littman soccer geometry:

- Grid: width 7 (`x` in `0..6`), height 5 (`y` in `0..4`).
- Player 0 starts left and scores through the right edge (`x = 7`); player 1
  starts right and scores through the left edge (`x = -1`).
- Goal mouth: middle rows `y in {1, 2, 3}`.
- Actions: `U, D, L, R` (no stay). Vertical moves clamp at the top/bottom walls;
  horizontal moves clamp except a ball carrier moving into the opponent goal.
- Collisions: if both players target the same square, the ball carrier takes it
  and possession flips to the other player; on a swap, players swap and
  possession flips. A carrier blocked by a stationary opponent stays put and
  loses the ball.
- Rewards: `+1` win, `-1` loss, `0` otherwise (zero-sum). Game ties after 100
  steps.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```
