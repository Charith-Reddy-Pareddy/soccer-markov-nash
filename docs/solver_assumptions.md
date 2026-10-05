# Assumptions made by the solvers and learners

[assumptions.md](assumptions.md) covers the *game rules*. This page covers
everything the solvers and learners assume on top of them. "Mine" means neither
the A10 text nor the group note asked for it; each row is a place a different
instruction could change a result.

## Exact solver

| # | assumption | where | status |
|---|---|---|---|
| S1 | The earlier experiments use **stationary** value iteration with `gamma = 0.9`; the state has no step counter, so the 100-step tie is not modelled. The undiscounted game is solved separately by backward induction. | `nash_q.py` | mine |
| S2 | Each stage game is a **zero-sum matrix game** solved for its minimax value (Shapley 1953). | `nash_q.py` | standard |
| S3 | Where a stage game has several equilibria, the reported policy is **one** of them; the value and `Q` are unique, the policy is not. | `degeneracy.csv` | mine |
| S4 | Pure vs. mixed is decided with a tolerance that scales with the matrix entries. | `numerics.py` | mine |

## Policy gradient (REINFORCE, A2C, PPO)

| # | assumption | where | status |
|---|---|---|---|
| S5 | Training uses the `+/-1` goal reward and `gamma = 0.9`, from the kickoff; episodes restart at the kickoff after a goal. Off-path states are rarely visited, but the evaluation scores **all** 2,380 states. | `policy_gradient.py`, `actor_critic.py` | mine |
| S6 | In the earlier experiments the 100-step tie is **not** enforced inside training episodes. A rollout is cut every 100 steps; the cut is bootstrapped by the critic (A2C, PPO) or given zero continuation (REINFORCE). | same | mine |
| S7 | The two players have **separate** policy and value networks. Mirror symmetry is measured (`mirror_gap`), not imposed. | same | mine |
| S8 | A2C/PPO settings (`lr 1e-3`, hidden 64, entropy 0.01, GAE `lambda 0.95`, clip 0.2, 4 PPO epochs, 2000 x 100 steps) are **untuned** and match REINFORCE's budget. A loss is not evidence against the algorithm. | `actor_critic.py` | mine |
| S9 | Win rates are Monte-Carlo estimates from the kickoff over 1,000 games, a tie is no goal in 100 steps, and the "best response" is exact in *discounted value*, not in win probability. | `winrate.py` | mine |
| S10 | The main environment is the **deterministic A10 board** (answered); the random move-order board is the earlier comparison. | `pg_finite.py` | answered |

## Exploring starts and DQN

| # | assumption | where | status |
|---|---|---|---|
| S19 | **Exploring starts** (`--explore`) restart every training episode at a uniformly random non-terminal state instead of the kickoff. The default is still the kickoff. | `policy_gradient.py`, `actor_critic.py` | mine |
| S20 | The Nash-DQN's *policy* is the minimax solution of its own predicted `Q` matrix at each state; ties are broken by the LP solver. | `pg_algos_and_fp.py` | mine |

## Fictitious play

| # | assumption | where | status |
|---|---|---|---|
| S11 | Both players update simultaneously, ties go to the lowest action index, and each starts on action `U`. | `fictitious_play.py` | mine |
| S12 | Matrix-level fictitious play (a side comparison; the professor's fictitious play is the PG scheme in `pg_finite.py`). Two versions are run: on the **exact stage matrices**, and **inside the Markov game** where `V` is rebuilt every sweep. The in-game version keeps one count vector per state; it does not use a learned best-response network. | same | mine |
| S13 | The value estimate is the midpoint of the bounds `min_j (pM)_j <= v <= max_i (Mq)_i` of the empirical mixes. | same | mine |

## Finite-horizon policy gradient

| # | assumption | where | status |
|---|---|---|---|
| S22 | The objective is discounted (`gamma = 0.9`) over 100 steps with a tie at the end; the network input is the state plus the remaining fraction of the horizon. | `pg_finite.py` | answered |
| S23 | A2C uses a 10-step bootstrapped advantage; PPO uses GAE with `lambda 0.95`, clip 0.2 and 4 epochs; REINFORCE uses plain discounted returns with no baseline. All share an entropy bonus of 0.01, `lr 1e-3`, 64 episodes per batch and 2,000 iterations, untuned. | `pg_finite.py` | mine |
| S24 | **Fictitious play** snapshots each network every 5 iterations; the opponent's average policy is realised by playing one uniformly drawn snapshot per episode, and the *reported* policy is the per-state average of the snapshots' action probabilities. Averaging behaviour per state is not the same as averaging whole strategies, so the reported policy is an approximation to the mixture that was trained against. | `pg_finite.py` | mine |
| S25 | Exploitability is the discounted finite-horizon best-response value of both players from the kickoff; the "best response" opponent in the win counts is that exact best response, optimal in discounted value, not in win probability. | `finite_horizon.py` | mine |

## Continuous actions and the dog game (provisional)

None of this has a specification yet. Everything here is a placeholder to get
something runnable, and is meant to be replaced.

| # | assumption | where | status |
|---|---|---|---|
| S14 | **Dog game.** A dog (pursuer) and a sheep (evader) move in the unit square `[0,1]^2`. Each step both choose a displacement `r(cos t, sin t)` with `r <= delta` (dog `0.06`, sheep `0.04`); positions are clipped to the square. The dog scores `+1` on capture (distance `<= 0.05`), and loses (`-1`) if the sheep survives 100 steps. Zero-sum. | `dog_game.py` | **invented** |
| S15 | **Angle-radius policy.** The angle `t` is von Mises, the radius is `delta * Beta`, both parameterised by the network. The two are sampled independently. | `dog_game.py` | **invented** |
| S16 | Dog and sheep start at fixed points `(0.2, 0.2)` and `(0.8, 0.8)`. | `dog_game.py` | **invented** |
| S17 | The three continuous best-response methods (bisection on the derivative, finite-difference gradient ascent, quadratic fit) are tested on **analytic** functions with a known maximiser, not on any learned `Q`. | `continuous_br.py` | mine |
| S18 | Dog-game results are a smoke test: there is no exact solution to compare against, so nothing is claimed about equilibrium. | `dog_game.md` | mine |
| S21 | The 10-angle DQN dog trains against a **fixed fleeing sheep** (runs straight away at full speed), always moves at full speed, and the 10 angles are equally spaced. | `dog_game.py` | **invented** |

## Answers received

The professor's replies to the group's questions, and what changed:

- **Environment.** Any board where the solver is correct and the comparison is meaningful; fewer mixed stage games is better this time. The main policy-gradient experiments now use the deterministic A10 board (no mixed stage games), and the random board is the earlier comparison.
- **Objective.** Policy gradient can only approximate a finite-horizon discounted reward, so both: `gamma = 0.9` **and** the 100-step horizon, with the remaining steps as a network input. This replaces S1 and S6 for the main experiment.
- **Fictitious play.** It is how policy gradient solves the game; there is no explicit game solving in it. The main fictitious-play runs therefore train each player by policy gradient against the average of the opponent's past policies. The matrix-level fictitious play below is kept only as a side comparison.
- **Win rate.** Repeated games, counting wins (this page also keeps the ties and losses).
- **Continuous actions.** The bisection, finite-difference and quadratic best-response methods are for DQN, not policy gradient. Policy gradient is adapted to continuous actions directly, which is what the dog-game angle-radius policy already does.

## Questions for the professor

Still open: 1, 2, 6, 8, 10, 11 and 12. Answered above: 3, 4, 5, 7 and 9.


1. **Dog game.** Pursuit-evasion in the plane? Who is the dog and who the sheep,
   what are the speeds, the arena, the capture or goal rule, the reward and the
   horizon? Is time discrete? (S14, S16)
2. **Polar policy.** Von Mises plus Beta, or a deterministic output plus noise?
   Is the radius cap the same for both players? (S15)
3. **Soccer variant.** The A10 deterministic board has 0 mixed stage games, the
   random move-order board has 94. Which should the learners be judged on? (S10)
4. **Discount.** Is `gamma = 0.9` acceptable, or should the learners see the
   undiscounted game with a 100-step timer and the step number as an input? (S1, S6)
5. **Win rate.** Is it `P(score first within 100 steps)`, and how is a tie
   counted? (S9)
6. **Symmetry.** Impose mirror equivariance (one network, a side flag), or only
   test it? (S7)
7. **Fictitious play.** Is the in-game version with per-state counts what the
   note means, or should the best response be a learned network? (S12)
8. **`Q*`.** Is it the per-state 4x4 matrix `Q(s, a0, a1)`?
9. **Continuous best response.** Are bisection, finite differences and the
   quadratic fit for the dog game now, or later? (S17)
10. **Evaluation states.** Compare with the exact solver on all states, or only
    those reachable from the kickoff? (S5)
11. **Degenerate equilibria.** Is matching the value enough, or should the policy
    match one particular equilibrium? (S3)
12. **A possible slip in the note.** With standard rock-paper-scissors payoffs the
    best response to `1/2 R + 1/2 S` is Rock (expected `+1/2`), not Paper.
