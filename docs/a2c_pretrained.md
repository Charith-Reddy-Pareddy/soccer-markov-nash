# A2C with an exact-value pretrained critic

The `a2c_pretrained` experiment implements the value-network procedure discussed
for policy gradient. It first fits a neural value network to the exact finite-
horizon solver values over the game's valid state and time pairs. Player 1's
targets use the negative of player 0's values. The fitted networks are then
frozen while the policy networks train.

For a sampled joint action, the policy update uses the exact solver's action
value and the frozen network's estimate as its baseline:

`A_i(t, s, a_0, a_1) = sign_i * Q*(t, s, a_0, a_1) - V_net,i(t, s)`

Here `sign_0 = +1` and `sign_1 = -1`. The value network does not receive policy-
training gradients. This separates the neural critic approximation from the
exact action-value lookup, so it tests the pretraining procedure without
presenting a finite training run as a learned equilibrium solver.

The existing `a2c_exact` option remains a distinct oracle comparison: it uses
the exact solver's `Q* - V*` advantage directly and does not train a neural
critic. Ordinary `a2c` learns its critic from rollout returns.

Run the new comparison with:

```sh
python scripts/pg_finite.py --algos a2c_pretrained --board a10
python scripts/pg_finite.py --algos a2c_pretrained --board random
```

The critic fit defaults to 10 complete passes through shuffled state-time
samples, in batches of 4,096. Adjust it with `--critic-pretrain-epochs` and
`--critic-pretrain-batch-size`. The resulting metrics are written to the same
experiment CSV format as the other finite-horizon PG runs.

After critic pretraining, REINFORCE and A2C make five optimization passes over
each newly collected rollout batch by default (`--updates-per-batch`). This
count is independent of critic pretraining. PPO continues to use its four
clipped-policy epochs. Multi-update REINFORCE/A2C runs use a separate
`_updates5.csv` output so they cannot overwrite or be confused with the
historical single-update results.
