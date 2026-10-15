# A2C with an exact-value pretrained critic

The `a2c_pretrained` experiment implements the value-network procedure discussed
for policy gradient. It first fits a neural value network to the exact finite-
horizon solver values over the game's valid state and time pairs. Player 1's
targets use the negative of player 0's values. The fitted networks are then
frozen while the policy networks train.

The actor update uses the sampled discounted return minus the frozen value
network's estimate as its baseline:

`A_i(t, s) = G_i(t) - V_net,i(t, s)`

Here `G_i(t)` is a Monte Carlo discounted return from sampled play, with
`sign_0 = +1` and `sign_1 = -1` applied to rewards. The value network does not
receive policy-training gradients. Exact action values are not queried during
actor updates. This separates the frozen state-value baseline from an oracle
action-value policy update.

The existing `a2c_exact` option remains a distinct oracle comparison: it uses
the exact solver's `Q* - V*` advantage directly and does not train a neural
critic. Ordinary `a2c` uses a learned critic with the repository's 10-step
bootstrapped advantage. The `a2c_mc` control uses a learned critic but the same
sampled full-return actor estimator as `a2c_pretrained`; comparing those two
isolates the baseline pretraining/freezing choice more closely.

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
