# Self-play and exploitability

`scripts/selfplay.py` plays the Nash Q policy against itself and measures how
exploitable various player-0 policies are.

## Nash vs. Nash from the kickoff

| move order | computed `V(kickoff)` | win / tie / loss | empirical discounted return (5 seeds) |
|---|---|---|---|
| deterministic | `0.000` | 0.00 / 1.00 / 0.00 | `+0.000 +/- 0.000` |
| coinflip | `0.000` | 0.00 / 1.00 / 0.00 | `+0.000 +/- 0.000` |
| random | `+0.150` | 0.65 / 0.00 / 0.35 | `+0.149 +/- 0.005` |

The deterministic and coinflip games are forced draws from the centre -- the
equilibrium play never scores and the game runs to the 100-step tie. In the
random game the initial carrier converts its `+0.15` edge into a ~2:1 win ratio,
and the empirical discounted return matches the value function within one
standard deviation over 5 seeds of 3000 games (`scripts/selfplay.py --seeds 5`).

## Exploitability

For a fixed player-0 policy, player 1 solves the induced MDP and best-responds.
The **duality gap** `V0_br(s0) + V1_br(s0)` is `0` exactly when both policies are
equilibria; for a one-sided policy, `V1_br(s0) + V(s0)` is how far above the
equilibrium value an optimal opponent can push (`soccer_nash.exploit`).

| player-0 policy | exploitability (deterministic game) |
|---|---|
| Nash Q policy | `0.0000` |
| uniform random | `0.39` |
| best response to the scripted opponent (Part 2) | `0.43` |

The Part 2 network crushes the scripted opponent (a 7-step win) but is *more*
exploitable than moving at random against an opponent that best-responds to it.
This is exactly the A10 competition's warning that "other students may
strategically submit a non-equilibrium policy": best-responding to one assumed
opponent is fragile, the Nash policy is the safe choice, and the
`pure`/`hybrid`/`mixed` solver produces it directly.
