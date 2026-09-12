# Player positions, drawn the way the meeting sketched them

At the research meeting the professor drew a stage game by hand: a small grid
of cells, each marked, with arrows chasing each other around the grid to show
that no cell is a stable outcome. `scripts/positions.py` (`make positions`)
turns that sketch into a real diagram, generated from the exact solver instead
of drawn free-hand, and pairs it with a picture of where the two players
actually are on the board when that stage game applies.

![Three cases: a pure state (one lit-up cell, no cycle), the meeting's own
L/R-indifference example, and a genuine 3-action mix -- each paired with the
board position and the stage game redrawn as a node-and-arrow
graph.](figures/png/positions.png)

## Reading the graph

Each node is one cell of the stage game (`row = the carrier's action`,
`col = the defender's action`), labelled with player 0's exact payoff. A blue
arrow points from a cell to the cell in the same column with the higher
payoff -- the carrier's own reason to switch rows. A green arrow points from a
cell to the cell in the same row with the *lower* payoff -- the defender's
reason to switch columns, since the defender minimises. Only the cells that
survive [iterated dominance](../soccer_nash/numerics.py) are drawn (`essential_
subgame`), so a 4x4 game collapses to whatever's actually load-bearing:

- **A pure state has exactly one node with no outgoing arrow.** That cell is
  the saddle; everything else eventually points to it. Nothing to guess.
- **A mixed state has arrows everywhere.** Every node points somewhere, so
  the arrows chase each other around the grid with no resting point -- the
  professor's picture, reproduced exactly.

## Case 2: the meeting's own example

> "when we move right and when we move left, there's an equal chance of me
> winning. That's why I am indifferent between the two."

State `(1, 1, 1, 0, 1)`: player 1 carries, and its equilibrium support is
**exactly `{L, R}`** -- no vertical option at all, so this isn't a
rock-paper-scissors-style best-reply cycle dressed up in a bigger matrix, it's
the plain case the meeting asked for: two actions that are genuinely equally
good against the defender's own mix (`D 18% / L 82%`), so there is no honest
reason to prefer one over the other. Per [numerics.md](numerics.md) §0, that
equal-payoff condition *is* the definition of a mixed equilibrium -- "the best
replies cycle" and "the played actions are indifferent" are the same fact
seen from two sides, not two different explanations.

## Case 3: three actions, not a two-cycle

State `(0, 0, 2, 0, 0)` -- the same far-from-goal 3-way mix from
[showcase.md](showcase.md) -- gets the graph treatment here to make the point
requested directly: the *matrix itself*, not a summary statistic, is the
output. Entropy (`1.109` bits here) is one number computed from the policy;
the graph and the printed matrix are the thing entropy is a lossy summary of.

## Reproduction

`python scripts/positions.py` prints the exact stage matrix and policy for
all three states and writes `figures/gallery/positions.svg`.
