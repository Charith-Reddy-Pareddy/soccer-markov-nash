# Reward and Q-matrix audit

The supervisor's question is justified: a zero matrix requires a check of the reward, transitions, continuation values and numerical precision. It cannot be justified merely by pointing to one self-loop. Also, state (3,4,4,4,0) does **not** have an all-zero matrix: its sparse-reward Q(D,L) is 0.6561 and Q(R,L) is -0.59049; its other 14 entries and minimax value are zero.

## The reward difference, verified from Jae's viewer

Source: [Jae's site](https://jwsong118.github.io/marlmarkov.html), [viewer implementation](https://jwsong118.github.io/soccer-mpe/soccer-viewer.js), [deterministic values](https://jwsong118.github.io/soccer-mpe/deterministic/data.js), retrieved September 28, 2026 (local time).

Both use gamma = 0.9. The default model here pays +1/-1 only at a terminal goal and zero otherwise. Jae's viewer also pays +1/-1 on terminal goals, without an extra bonus, but pays on **every nonterminal next state**:

- If player 0 carries: r0 = 0.005 + 0.005*x0_next.
- If player 1 carries: r0 = -(0.005 + 0.005*(6-x1_next)).
- Player 1's reward is -r0.

This rewards possession and absolute progress, even while standing still. It is neither a reward for the change in distance nor the repository's existing pre-state, unconditional `StepProgressionBonus`. The comparison uses an explicit opt-in game so exact, DQN and PG all receive identical rewards. The default game's objective is unchanged.

At (3,4,4,4,0), (U,U) retains possession at x=3, paying 0.020. In Jae's computed equilibrium V=0.2, so Q(U,U)=0.020+0.9*0.2=0.2. The pair (U,L) transfers possession at x1=4, paying -0.015, followed by V=-0.15: Q=-0.015+0.9*(-0.15)=-0.15. These are discounted reward units, **not win probabilities**. Matching reward, discount, transitions and termination is necessary before comparing solvers.

## What the tests certify

The audit enumerates all 2,380 states and all 16 joint actions. It reconstructs raw Q=r+gamma*V(next), with zero terminal continuation, and checks the computed policies’ security bounds against V, allowing mixed strategies for dense rewards. A zero equilibrium gap plus a small Bellman residual certifies the minimax fixed point for this discounted model. This checks profitable deviations too; a self-loop-only argument does not. Sparse results contain 514 exactly all-zero matrices and 1,014 zero-value states. Floating-point linear solving is numerical, not symbolic algebra; the report records residuals before formatting. As a stronger independent certificate, `python scripts/certify_sparse_rational.py` converts the sparse candidate to rational powers of 9/10, then checks all 38,080 actual transitions and the complete maximin/minimax Bellman equalities with exact fractions. All equalities hold exactly and the all-zero count remains 514. The numerical candidate is accepted only after those rational checks; display precision plays no role.

A flat zero matrix means every immediate action pair leads to zero reward and zero-valued optimal continuation. It does not mean all future policies draw, that nobody can score against mistakes, or that the state is terminal. These results concern this deterministic rule, not a general claim about Littman's random-order soccer game.

## Actual defects corrected

1. Both independent and shared policy-gradient collectors discarded terminal flags while restarting inside a rollout. Discounted returns then included rewards from a different game. For rewards [0,+1,-1] with the second and third steps terminal, the correct returns at gamma=.9 are [.9,1,-1]; the old calculation gave [.09,.1,-1]. Collectors now preserve terminal masks, and both trainers reset the backward return at each terminal goal. Continuing `scoring="rate"` games retain their continuing returns. Three new regression cases failed before the fix and pass after it.
2. The edge-case PDF called the screenshot's matrix all-zero and treated one self-loop as proof of a zero value. It also conflated old PG behavior with verified state visitation. Those explanations have been replaced with the full-board certificate and measured experiments. No visitation claim is inferred from a bad action alone.

3. Fitted DQN recomputed identical minimax targets every epoch while its target network remained frozen. Targets are now cached until synchronization (80 evaluations rather than 400 with the default five-epoch interval). A seeded before/after comparison of all network predictions and losses was bit-for-bit identical; a regression checks refresh frequency and the original loss trace. This fixes redundant work, not the mathematical objective.

## How to interpret DQN and PG

The repository's DQN is model-based fitted minimax-Q iteration, using full transition expectations and a target network. It approximates a 4x4 equilibrium Q matrix. The audit trains from random initialization; it does not train on the exact answer. A supervised fit to exact Q is a representation check, not independent validation.

REINFORCE outputs action probabilities, **not** 16 joint-action Q values. The audit evaluates its two learned stationary policies by solving (I-gamma*P_pi)V_pi=R_pi, then constructs Q_pi=r+gamma*V_pi(next). This is the value of following those policies, not an equilibrium Q estimate unless those policies form an equilibrium. Finite runs may disagree with exact Q because of approximation, optimization and coverage. No equality or convergence is promised from a finite neural experiment. Two fixed seeds, 400 DQN epochs and 1,000 PG iterations with 100-step rollouts are recorded in `experiments/reward_q_audit.json`, for each reward model; this is a bounded reproducibility check, not an exhaustive hyperparameter study.

## Research context

[Littman (1994)](https://cs.uwaterloo.ca/~klarson/teaching/W06-886/papers/Littman94.pdf) gives the minimax Markov-game backup and stochastic soccer setting. Its theory explains why values depend on reward and optimal continuation; it does not prove this project's deterministic transitions are the same game.

[Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) distinguish arbitrary bonuses from potential-based shaping. With F=gamma*Phi(next)-Phi(current), bounded discounted returns telescope (terminal Phi=0), shifting each policy's value by -Phi(current). Jae's recurring possession/progress bonus should not be assumed policy-invariant; it defines a different objective.

[Wei et al. (2021)](https://arxiv.org/abs/2102.04540) establish convergence for a specialized optimistic actor-critic procedure in competitive Markov games. That result is not a convergence guarantee for this repository's vanilla self-play REINFORCE.

Reproduce: `python scripts/reward_q_audit.py`; then `python -m pytest -m ''`. The optional `--reference` file holds values and transitions extracted from Jae's public viewer for the external comparison. Source hashes: viewer SHA256 `99198daeca409a08a41f4a0ecd58bf4f4fcd4fd3ce3b20db08b5de14e99e70cd`; data SHA256 `cdf1428d56fd219a0708e401ec5537fed6cee246083f0e54f2349eee9a39e407`.


External comparison result: all **38,080 transitions/rewards match**. The re-solved dense model differs from Jae's eight-decimal V data by at most 5.0000002e-9; reconstructed Q differs by at most 4.5000002e-9. Its 62 mixed states have maximum equilibrium gap 2.22e-16 and Bellman residual 1.67e-16. The screenshot difference is at most 5e-7, consistent with six-decimal display rounding. This establishes the reward discrepancy directly across the entire deterministic board.

To reproduce the external comparison, download the two public JavaScript sources linked above, then run `node scripts/extract_reward_reference.cjs viewer.js data.js reference.json` and `python scripts/reward_q_audit.py --reference reference.json`. Run `python scripts/write_reward_report.py` to rebuild the HTML appendix from the recorded experiment and external certificate; print `docs/a10_cases.html` to regenerate its PDF.

## Completed validation

All 422 Python tests passed with slow tests enabled (`python -m pytest -q -m ''`), including terminal-return, reward, screenshot and frozen-target regressions. All 6 site tests, site lint and production build passed. The refreshed explorer contains 11,480 valid PG policy records across six boards; all saved DQN records were checked unchanged.

Two-seed measurements (MAE across all 38,080 Q entries):

| Reward | Seed | DQN vs equilibrium Q | Evaluated PG Q vs equilibrium Q |
|---|---:|---:|---:|
| Sparse | 0 | 0.140408 | 0.185159 |
| Sparse | 1 | 0.145538 | 0.214319 |
| Jae possession/progress | 0 | 0.115229 | 0.154976 |
| Jae possession/progress | 1 | 0.107093 | 0.140485 |

These learned runs do not reproduce exact Q; full-board maximum errors range from 1.043 to 1.286 for DQN and 0.749 to 0.939 for evaluated PG. They are measured limitations, not a substitute for the equilibrium certificates. Source inspection and matching all transitions/rewards resolve the cross-implementation discrepancy independently of neural convergence.
