# A10 competition: the equilibrium networks

The competition is a round robin where "other students may strategically submit
a non-equilibrium policy." A best response to any particular guess about the
field is fragile (see `docs/selfplay.md`: best-responding to the scripted
opponent is more exploitable than random play). The **Nash Q policy** is the
principled submission: it cannot be beaten below the game value by any opponent,
and it still takes every forced win an opponent offers.

## Pipeline (`scripts/a10_competition.py`)

1. Solve the deterministic game with Nash Q iteration (9 sweeps, pure
   equilibrium, `V(kickoff) = 0`).
2. For every state, take each player's **security-optimal action set** -- the
   rows whose worst case equals the maximin value, the columns whose best case
   equals the minimax value (`NashQIteration.optimal_action_masks`). ~1350 of
   2380 states have all four actions optimal (the neutralised dead zones);
   ~570 have a unique optimal action.
3. Fit a bias-free `5 -> 99 -> 99 -> 4` network to each side with partial-label
   training (any optimal action counts):
   - Network First (player 0), Network Second (player 1).
4. Report residual exploitability and write the weights in the A10 matrix
   format.

```bash
python scripts/a10_competition.py --out results/
```

Output `results/a10_competition.txt` holds both networks (three matrices each,
separated by `-----`), git-ignored -- every student trains their own.

## The approximation is only as safe as its worst state

The *exact* Nash policy has exploitability 0. A network approximation does not
(`scripts/a10_competition.py --seeds 5`, `experiments/a10_competition_seeds.csv`):

| network | plays an optimal action | exploitability (from the A10 kickoff) |
|---|---|---|
| Network First (player 0) | `0.994 +/- 0.000` | `0.19 +/- 0.02` |
| Network Second (player 1) | `1.000 +/- 0.000` | `0.00 +/- 0.00` |

A single state where the network prefers a losing move is enough for a
best-responding opponent to gain from the right starting position. Network
Second learns its (larger) optimal-action set exactly; Network First does not,
and its exploitability is stable across seeds -- it is a capacity / fit limit,
not bad luck. (The exact figure depends on the kickoff, since it is measured
there.) The equilibrium policy is the safe submission regardless.
