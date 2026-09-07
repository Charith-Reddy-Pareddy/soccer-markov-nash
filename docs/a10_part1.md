# A10 Part 1: successor states and rewards

Part 1 asks you to reproduce the environment and compute the win/lose rewards:

- **Q1 / Q3** -- for a given state `(x0, y0, x1, y1, b)`, the successor state
  after each of the 16 joint actions
  `UU UD UL UR DU DD DL DR LU LD LL LR RU RD RL RR`
  (first letter = player 0, second = player 1). A win is `(-1, -1, -1, -1, p)`.
- **Q2 / Q4** -- the reward of each of those 16 transitions: `+1` win, `-1`
  loss, `0` otherwise.

## Usage

```bash
python scripts/a10_part1.py 3,2,4,2,0
python scripts/a10_part1.py 3,2,4,2,0 --goal-rows 1 2 3 --width 7 --height 5 --player 0
```

The script prints the 16 successor lines (each tagged with its action) and then
the single reward line. `soccer_nash.a10.format_successors` / `format_rewards`
give the same strings without the tags.

## Notation

The A10 page writes positions 1-indexed (`player 1`, `player 2`) but the ball
flag `b` and the winner `p` 0-indexed. This repo is 0-indexed throughout:
`player 0` == A10's "player 1", `player 1` == A10's "player 2". `b = 0` means
player 0 carries the ball.

## Assumptions

- The public A10 page redacts the ID-specific start position and goal rows, so
  `--width`, `--height` and `--goal-rows` are configurable. Defaults are the
  Littman layout: 7x5 board, goal mouth on rows `{1, 2, 3}`, player 0 scoring
  through the right edge and player 1 through the left.
- Part 1 uses the deterministic move order (the A10 collision rules).
- A carrier that moves into a standing opponent is treated as "trying to occupy
  the same square": it stays put and loses the ball. The A10 text does not spell
  this sub-case out; this is the reading consistent with its two stated rules.
- The reward line defaults to player 0's perspective; use `--player 1` to flip.
