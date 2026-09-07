# A10 Part 2: policy network vs. the scripted opponent

Part 2 asks for a policy network (Q7) and a winning state sequence (Q8) against
the scripted opponent:

- **no ball** -> move toward the ball carrier;
- **has ball** -> move toward its own goal;
- horizontal direction first, then vertical.

## Pipeline

`scripts/a10_part2.py` runs three steps:

1. **Exact best response.** With the opponent's move fixed, the game is a
   deterministic single-agent MDP. Value iteration
   (`soccer_nash.best_response.BestResponse`) gives the optimal value function
   and a greedy policy in ~15 sweeps.
2. **Imitation.** A bias-free `5 -> 99 -> 99 -> 4` ReLU/softmax network
   (`soccer_nash.mlp.MLP`) is trained to match the optimal action at every
   state. Training uses partial labels: at the ~50% of states with several
   equally optimal actions, any of them counts, which lifts agreement from
   ~90% to ~99%.
3. **Q8 rollout.** The trained network (argmax) plays player 0 against the
   scripted opponent from the kickoff. The resulting trajectory is, by
   construction, consistent with the network; the script asserts it is a win
   in <= 100 steps before writing anything.

```bash
python scripts/a10_part2.py --out results/ --hidden 99 --gamma 0.9 --epochs 2500
```

Outputs (git-ignored -- every student trains their own):

- `results/a10_q7_weights.txt` -- three matrices separated by `-----`, rows on
  their own lines, columns comma-separated (`5 x h1`, `h1 x h2`, `h2 x 4`).
- `results/a10_q8_trajectory.txt` -- one state per line, ending in
  `-1,-1,-1,-1,0`.

## Notes

- **Bias-free** to match the A10 weight format (three matrices, no bias
  vectors). Inputs are the raw integers the grader feeds; the trainer applies an
  internal `[1/6, 1/4, 1/6, 1/4, 1]` input scale for conditioning and folds it
  back into the first matrix on export, so the saved weights behave identically
  on raw inputs.
- Best-response `V(kickoff) = 0.53` (`gamma^6`, a 7-step win); over 5 seeds
  (`scripts/a10_part2.py --seeds 5`) the network plays an optimal action at
  `0.990 +/- 0.000` of states and wins the Q8 rollout every time, in 7 steps.
- The opponent is deterministic, so the best response wins outright; there is no
  need for the Nash machinery here (that is for self-play, not a fixed
  opponent).
