# Second review -- 18 points

Tracking the second round of feedback (the geometry-model / research-framing
review). The reviewer's own priority order is in point 16.

| # | point | status |
|---|---|---|
| 1 | Geometry model as a *discovery tool*, not the final result | done -- [templates.md](templates.md) turns the tree's split into a mechanism |
| 2 | Central question -> "which geometries force mixing, can they be characterized analytically"; two contributions (algorithmic + structural) | done -- report + README lead with the two contributions |
| 3 | Don't overclaim the one-cell-goal result; proof is a long-term goal | done (first review) -- [assumptions.md](assumptions.md) Claim C |
| 4 | Board sweep -> phase diagram: mixed / reachable vs width x height x goal width | pending -- experiment run |
| 5 | Record seeds; run seeds 0-4; report mean +/- std for stochastic experiments | pending -- experiment run |
| 6 | Runtime: median of >=5 runs; report LP-calls/state and fraction of states needing LP | pending -- experiment run |
| 7 | "25x" wording: keep LP-solve reduction and wall-clock speedup separate | done -- report RQ2 + README say so explicitly |
| 8 | Explicit stage-game taxonomy (exact / numerically-indistinguishable / strict / degenerate / genuine mixed) | done -- `classify_stage_game` docstring + report RQ4 table |
| 9 | Strengthen "matching pennies": compute each 2x2 support matrix, test the crossing inequalities | **done** -- `templates.matching_pennies_pattern`, 68/94 verified |
| 10 | Name actions consistently: {U,D,L,R} -> advance/climb/block/... per template, show real matrices | **done** -- `templates.describe_action`, [templates.md](templates.md) |
| 11 | Remove the dog/debate game | done (first review) |
| 12 | DQN phase: exact hybrid as ground truth, full comparison table | partial -- table exists; multi-seed rides on point 5 |
| 13 | Paper structure not centered on A10 Part 2 | done -- Problem / Method / RQs / Results / Mechanism / Limitations / Algorithm; A10 Part 2 is section 8 (secondary validation), Parts 1/competition are separate docs |
| 14 | Title -> "When Does a Soccer Markov Game Need Mixed Strategies?" | done -- report + README + HTML title |
| 15 | Novelty = the geometric characterization of unavoidable mixing | done -- it is contribution 2, and [templates.md](templates.md) |
| 16 | Priority order | this table |
| 17 | Canonical-state analyzer: 2380 states -> equivalence classes -> few geometric templates + stage matrix per class | **done** -- `soccer_nash/templates.py`, [templates.md](templates.md): 94 -> 47 pairs -> 8 templates |
| 18 | Characterize *why* multi-cell goal + interception geometry forces mixing | **done** -- [templates.md](templates.md) "Why the mix is unavoidable" |

## This pass

- **1, 9, 10, 17, 18** -- `soccer_nash/templates.py` + `scripts/templates.py` +
  `docs/templates.md`: the 94 no-pure-saddle states reduce to **8 geometric
  templates**; the **four with a 2x2 equilibrium support are all genuine
  matching pennies** (best replies cross both ways), covering **68 of 94**
  states. The remaining 26 are borderline (a near-pure saddle) or a 3x3 mix.
- **2, 7, 8, 13, 14, 15** -- one editing pass on `docs/report.md` + README:
  retitled, lead with the two contributions, added the five-way stage-game
  taxonomy to RQ4, split "LP-solve reduction" from "wall-clock speedup".

The precise analytic condition for an unavoidable mixed stage game:

> stochastic (½–½) resolution order **and** a goal mouth wider than one cell
> **and** the defender able to contest the carrier's forward cell but not cover
> every scoring lane in one move.

## Still to do

**4 / 5 / 6 / 12** -- one measurement pass: re-run the sweeps recording seeds
and repeats, build the goal-width x board-geometry heatmap, report runtime as
medians and add LP-calls-per-state. The board sweep alone is ~10 min, so this
is its own change.
