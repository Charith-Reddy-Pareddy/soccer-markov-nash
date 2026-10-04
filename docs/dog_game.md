# Dog and sheep: continuous actions (provisional)

The group note's second half is a continuous game with a dog and a sheep, a policy that outputs an angle and a radius, and best responses found by bisection, finite differences or a quadratic fit. No definition of the game exists yet, so the game here is a placeholder. Every choice is listed in [solver_assumptions.md](solver_assumptions.md) (S14-S18), and none of it should be read as the intended game.

## What was built

- `soccer_nash/dog_game.py` -- the game (unit square, dog speed 0.06, sheep speed 0.04, capture within 0.05, 100 steps), the angle-radius policy (von Mises angle, Beta radius), and self-play PPO for both players on the sparse terminal reward.
- `soccer_nash/continuous_br.py` -- the three best-response methods for one continuous action. They are tested on analytic functions with a known maximiser, including a clipped boundary peak and the best reply to a fixed opponent action.

## Result

Self-play PPO, 300 iterations of 64 games, two seeds, 500 games per row, deterministic (mean) actions. The capture rate is the dog's:

| matchup | seed 0 | seed 1 |
|---|---|---|
| random dog vs. random sheep | 0.00 | 0.00 |
| straight-line dog vs. trained sheep | 1.00 (19 steps) | 1.00 (19 steps) |
| trained dog vs. trained sheep | 1.00 (33 steps) | 1.00 (27 steps) |
| trained dog vs. random sheep | 0.34 | 0.19 |
| random dog vs. trained sheep | 0.00 | 0.00 |

The trained dog catches the sheep it trained against, but slowly: a plain straight-line chase needs 19 steps, the learned dog 27-33. It catches a random sheep only 19-34% of the time, so what it learned is specific to its trained opponent, not a general pursuit. The faster dog should win in any case, so the trained sheep is not evidence of a good escape strategy.

Nothing here says anything about equilibrium. There is no exact solution to compare against, and the dog's speed advantage means the right answer to this placeholder game is probably not interesting.

## Reproducing

```
python scripts/dog_game.py --seeds 2     # ~20 min, writes experiments/dog_game.csv
```
