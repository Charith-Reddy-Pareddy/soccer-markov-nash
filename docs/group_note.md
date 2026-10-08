# The group note, item by item

What the handwritten note asks for, and where it is covered. "Provisional" means the game or choice behind it is an assumption ([solver_assumptions.md](solver_assumptions.md)).

| item in the note | status | where |
|---|---|---|
| DQN and PG on soccer, compared with `Q*` | done | [neural.md](neural.md), [policy_gradient.md](policy_gradient.md) |
| REINFORCE, A2C, PPO to replicate the exact solver | done on the A10 board, discounted over 100 steps, with the step count as input; none replicates it | [policy_gradient.md](policy_gradient.md) |
| network: state in, softmax over actions out | done, one per player | `policy_gradient.py` |
| check symmetry | measured in the main experiment (mirror gap 0.32-0.81 for the learners, 0 for the exact solution), not imposed | `finite_horizon.py` |
| win rates against random, Nash and best response, next to the exact solver | done for the main experiment, for REINFORCE, A2C, PPO and DQN (counts of wins, ties and losses over 1,000 games) | [policy_gradient.md](policy_gradient.md) |
| best-response dynamics do not reach the equilibrium (rock-paper-scissors) | done, cycles | `fictitious_play.py` |
| fictitious play reaches it | done as policy gradient against the average of the opponent's past policies (no game solving). The matrix-level and Markov-game versions are kept as a side comparison | [policy_gradient.md](policy_gradient.md) |
| `br(1/2 R, 1/2 S)` | Rock | [policy_gradient.md](policy_gradient.md) |
| the `1/3, 1/3, 1/3` mixed state | 8 three-way mixes among 87 mixed states | [policy_gradient.md](policy_gradient.md) |
| dog game, DQN with one output per angle | provisional | [dog_game.md](dog_game.md) |
| dog game, angle and radius policy output | provisional | [dog_game.md](dog_game.md) |
| continuous best response: bisection, finite difference, quadratic fit | for DQN; done on analytic functions and a fitted network | `continuous_br.py` |
| train a network to approximate a function, then find its extremum | done in a test | `tests/test_continuous_br.py` |
| closed-form solution of the "R game" | not done: unclear what game this is (question 13) | -- |
| `Q(s, a1, a2)` with both actions as inputs | not done: crossed out in the note | -- |
| "# time f is activated" (side quest) | not done: unclear what is counted (question 14) | -- |

On the note's "NE ??? 1/2": in the deterministic A10 game the exact equilibrium ties against itself every game (value 0), so a win rate against it tops out at a tie, not one half. On the random move-order board the exact row player wins 66% against the exact column player, mostly because it starts with the ball.
