# The dog / debate game

`soccer_nash/dog_game.py`, `scripts/dog_game.py`, `tests/test_dog_game.py`.

The research group's second game: two players move on a grid, each pulling a
"dog" toward their own house. The dog sits at a weighted average of the two
players' positions,

    dog = w_a * pos_a + w_b * pos_b        (w_a + w_b = 1)

and player `a` is rewarded by `-dist(dog, house_a)`, player `b` by
`-dist(dog, house_b)`. In 2D the two distances are not complementary, so the
game is **general-sum**. The closed-form Nash equilibrium (from the meeting):
both players park in their own corners, and the dog settles at

    dog* = w_a * house_a + w_b * house_b.

## Why it is here

It is a general-sum Markov game with a *known* answer, so it validates
`solve_markov_game` (`markov_game.py` + support enumeration) end to end.

```
python scripts/dog_game.py
```

| variant | states | sweeps | support-enum states | dog at equilibrium | closed form |
|---|---|---|---|---|---|
| 1D (zero-sum), 5-cell line | 25 | ~90 | 0 | (2.0, 0.0) | (2.0, 0.0) |
| 2D general-sum, 4x4, w_a = 0.5 | 256 | ~165 | 0 | (1.5, 1.5) | (1.5, 1.5) |
| 2D general-sum, 4x4, w_a = 0.75 | 256 | ~169 | 0 | (0.75, 0.75) | (0.75, 0.75) |

Rolling the solved policy out from any start converges to the corners and the
dog to `dog*` exactly.

## Findings

- **Every stage game of the dog game has a pure Nash equilibrium** -- support
  enumeration is never triggered, even in the general-sum 2D case. The pure-first
  strategy that pays off for A10 soccer pays off here too.
- Many states have *multiple* pure equilibria (220 of 256: from an interior
  cell, several moves reduce the distance equally). For a general-sum game the
  selection rule matters; the "largest sum of values" rule (`select="largest_sum"`)
  keeps the players coordinated.
- The asymmetric-weight runs confirm the solver tracks `dog*` as `w_a` changes,
  not just the symmetric midpoint.
