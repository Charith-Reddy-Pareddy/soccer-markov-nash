# Do the players actually visit the mixed states?

`RQ2` reports that the hybrid solver calls the LP on **3.95%** of states -- the
94 no-pure-saddle states out of 2380. That counts the whole state space. The
professor's framing was that a stochastic transition and a mixed policy both
live in the **occupancy distribution**, so the sharper question is: how much of
the *equilibrium path* runs through the mixed region?

`soccer_nash/occupancy.py` computes the discounted state-visitation
distribution under the Nash policy from the kickoff -- an exact linear fixpoint,
`d = (1−γ)·μ₀ + γ·Pᵀπ·d`, not sampling. `scripts/occupancy.py` (`make
occupancy`) runs it on the 7×5 board, 3-cell goal, random order, `γ = 0.9`.

## The path is narrow, and it goes through the mixed region

| | value |
|---|---|
| states reachable under the Nash policy | **456 / 2380** |
| no-pure-saddle states | 94 (3.95% of all) |
| equilibrium time spent in mixed states | **41.4%** |
| their share of the *visited* states | 20.6% |
| concentration on the path | **2.0×** over-represented vs. visited, **10×** vs. all states |

Four of the eight most-visited states are no-pure-saddle. The carrier's optimal
trajectory from the kickoff -- march down the centre rows toward the attacking
goal -- passes directly through the band of contested cells where the stage game
is matching pennies. The mixed states are not a curiosity in a corner of the
state space; they are where the game is decided.

`docs/figures/png/occupancy.png` shades each carrier cell by equilibrium
time and marks the cells that hold a mixed state -- the hot cells and the marked
cells coincide.

## Why this matters for the algorithm

The "3.95% of states need the LP" number is the right one for a *worst-case*
per-sweep cost (you must check every state). But it understates how often the
mixed structure is *load-bearing*: a player who refuses to mix leaks value at
~40% of the moments that actually occur under equilibrium play, which is the
practical reason the pure-first hybrid keeps the LP fallback rather than settling
for the maximin bound ([mixing.md](mixing.md)).
