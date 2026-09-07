# Second review -- 18 points

Tracking the second round of feedback (the geometry-model / research-framing
review). The reviewer's own priority order is in point 16.

| # | point | status |
|---|---|---|
| 1 | Geometry model as a *discovery tool*, not the final result | done -- [templates.md](templates.md) turns the tree's split into a mechanism |
| 2 | Central question -> "which geometries force mixing, can they be characterized analytically"; two contributions (algorithmic + structural) | done -- report + README lead with the two contributions |
| 3 | Don't overclaim the one-cell-goal result; proof is a long-term goal | done (first review) -- [assumptions.md](assumptions.md) Claim C |
| 4 | Board sweep -> phase diagram: mixed / reachable vs width x height x goal width | done -- `scripts/phase_diagram.py`, `experiments/phase_diagram.csv`, heatmap in RQ3. Goal width 1 -> 0 on all 16 boards; wider -> 3-12%. Every state is reachable so reachable == all. |
| 5 | Record seeds; run seeds 0-4; report mean +/- std for stochastic experiments | done -- DQN (5 seeds, `nash_dqn_seeds.csv`), self-play return (+0.149 +/- 0.005), Limitations "Reproducibility" note |
| 6 | Runtime: median of >=5 runs; report LP-calls/state and fraction of states needing LP | done -- `scripts/benchmark.py`; RQ2 table now leads with LP calls / states-needing-LP (3.95%), wall clock is a median labelled machine-dependent |
| 7 | "25x" wording: keep LP-solve reduction and wall-clock speedup separate | done -- report RQ2 + README say so explicitly |
| 8 | Explicit stage-game taxonomy (exact / numerically-indistinguishable / strict / degenerate / genuine mixed) | done -- `classify_stage_game` docstring + report RQ4 table |
| 9 | Strengthen "matching pennies": compute each 2x2 support matrix, test the crossing inequalities | **done** -- `templates.matching_pennies_pattern`, 68/94 verified |
| 10 | Name actions consistently: {U,D,L,R} -> advance/climb/block/... per template, show real matrices | **done** -- `templates.describe_action`, [templates.md](templates.md) |
| 11 | Remove the dog/debate game | done (first review) |
| 12 | DQN phase: exact hybrid as ground truth, full comparison table | done -- report section 10: value (max + mean), action agreement, pure/mixed classification, exploitability, convergence, runtime, all mean +/- sd over 5 seeds |
| 13 | Paper structure not centered on A10 Part 2 | done -- Problem / Method / RQs / Results / Mechanism / Limitations / Algorithm; A10 Part 2 is section 8 (secondary validation), Parts 1/competition are separate docs |
| 14 | Title -> "When Does a Soccer Markov Game Need Mixed Strategies?" | done -- report + README + HTML title |
| 15 | Novelty = the geometric characterization of unavoidable mixing | done -- it is contribution 2, and [templates.md](templates.md) |
| 16 | Priority order | this table |
| 17 | Canonical-state analyzer: 2380 states -> equivalence classes -> few geometric templates + stage matrix per class | **done** -- `soccer_nash/templates.py`, [templates.md](templates.md): 94 -> 47 pairs -> 8 templates |
| 18 | Characterize *why* multi-cell goal + interception geometry forces mixing | **done** -- [templates.md](templates.md) "Why the mix is unavoidable" |

## Earlier passes

- **1, 9, 10, 17, 18** -- `soccer_nash/templates.py` + `scripts/templates.py` +
  `docs/templates.md`: the 94 no-pure-saddle states reduce to **8 geometric
  templates**; the **four with a 2x2 equilibrium support are all genuine
  matching pennies** (best replies cross both ways), covering **68 of 94**
  states. The remaining 26 are borderline (a near-pure saddle) or a 3x3 mix.
- **2, 7, 8, 13, 14, 15** -- editing pass on `docs/report.md` + README:
  retitled, lead with the two contributions, added the five-way stage-game
  taxonomy to RQ4, split "LP-solve reduction" from "wall-clock speedup".

The precise analytic condition for an unavoidable mixed stage game:

> stochastic (½–½) resolution order **and** a goal mouth wider than one cell
> **and** the defender able to contest the carrier's forward cell but not cover
> every scoring lane in one move.

## Measurement pass (4, 5, 6, 12)

- **4** `scripts/phase_diagram.py` -> `experiments/phase_diagram.csv` (46 configs,
  ~8 min) + `docs/figures/phase_diagram.svg` heatmap. Result: goal width is a
  binary gate -- **all 17 one-cell-goal configs have 0 mixed states; all 29
  wider-goal configs have some**, at a 3-12% fraction nearly independent of
  board size. `reachable_states` confirms the whole legal space is reachable, so
  mixed/reachable = mixed/all.
- **5** seeds 0-4 everywhere randomness bites: `scripts/nash_dqn.py --seeds 5`
  (`nash_dqn_seeds.csv`), `scripts/selfplay.py --seeds 5`. Reported mean +/- sd.
- **6** `scripts/benchmark.py`: median of 5 repeats; RQ2 table reordered to lead
  with LP calls and the reproducible "3.95% of states need an LP", wall clock
  demoted to an indicative median.
- **12** report section 10 is now a 6-row exact-vs-DQN table over 5 seeds.

## All 18 done. Open (long-term, not part of this review): a formal proof of
"one-cell goal => pure everywhere".
