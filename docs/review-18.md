# Second review -- 18 points

The geometry-model / research-framing review. Each row names the concrete
artifact that closes the point.

| # | point | where it lives |
|---|---|---|
| 1 | Geometry model as a *discovery tool*, not the final result | `docs/geometry.md` says so explicitly; the tree points at the condition, `templates.md` + `proof.md` turn it into a mechanism and a partial theorem |
| 2 | Central question -> "which geometries force mixing, can they be characterized analytically"; two contributions (algorithmic + structural) | `docs/report.md` and `README.md` open with the research question and the two contributions; "the two meet in RQ2" line |
| 3 | Don't overclaim the one-cell result; proof is a long-term goal | `docs/assumptions.md` Claim C / C'; `docs/proof.md` is careful about what is and isn't proved |
| 4 | Board sweep -> phase diagram: mixed / reachable vs width x height x goal width; compare board width vs goal width | `scripts/phase_diagram.py` (all boards `w<=11, h<=9, w*h<=45`, every goal width), `experiments/phase_diagram.csv`, `figures/phase_diagram.svg`; `--analyze` does the variance decomposition -- goal width (1 vs >=2) perfectly predicts *whether* mixing occurs, board area explains the residual *fraction* |
| 5 | Record seeds; run seeds 0-4; mean +/- sd for stochastic experiments | `scripts/nash_dqn.py --seeds` (`nash_dqn_seeds.csv`), `scripts/a10_part2.py --seeds`, `scripts/a10_competition.py --seeds` (`a10_competition_seeds.csv`), `scripts/selfplay.py --seeds`; report `&sect;10` Reproducibility note |
| 6 | Runtime: median of >=5 runs; report LP-calls/state and fraction of states needing LP | `scripts/benchmark.py` (median of 5); report RQ2 table leads with LP calls, states-needing-LP (3.95%), LP/state/sweep; wall clock is a labelled median |
| 7 | "25x" wording: LP-solve reduction != wall-clock speedup, consistently | report RQ2 ("Two numbers, kept separate" + "Do not conflate..."), README, `findings.md`, `selfplay.md` all use the seeded / reproducible numbers |
| 8 | Explicit stage-game taxonomy (exact / numerically-indistinguishable / strict / degenerate / genuine mixed) | `classify_stage_game` docstring (all five); report RQ4 table; forward-referenced from Method (`&sect;2`) |
| 9 | Strengthen "matching pennies": compute each 2x2 support matrix, test the crossing inequalities | `templates.matching_pennies_pattern` (checks `a>=c != b>=d` and `a<=b != c<=d`, returns `p*`, `q*`, value); `templates.md` shows all four 2x2 games and calls them "strategically equivalent to matching pennies" |
| 10 | Name actions consistently: {U,D,L,R} -> advance/climb/block/... per template; show the matrices | `templates.describe_action`; `templates.md` "The templates" table + Template 2 walk-through (state -> Q matrix -> support -> action names -> mixed probabilities) + the other three MP games |
| 11 | Remove the dog/debate game | done (first review); `grep` finds no `dog` references |
| 12 | DQN phase: exact hybrid as ground truth, full comparison table | report `&sect;8`: value (max + mean), action agreement, pure/mixed classification, exploitability, convergence (epochs to MSE plateau), runtime -- all mean +/- sd over 5 seeds; `nash_dqn_seeds.csv` |
| 13 | Paper structure not centered on A10 Part 2 | report reordered: Problem / Game+Nash-Q / **Algorithm** / RQs / Results / Mechanism / Secondary validation / Neural / Open questions / Limitations. A10 is one bullet in Secondary validation; Parts 1/2/competition are separate docs |
| 14 | Title -> "When Does a Soccer Markov Game Need Mixed Strategies?" | report `.md` / `.html` `<title>` / `<h1>`, README |
| 15 | Novelty = the geometric characterization of unavoidable mixing | it is contribution 2; `templates.md` "Why the mix is unavoidable"; the "two meet in RQ2" line ties it to the algorithm |
| 16 | Priority order | this table; items 1-7 of the reviewer's list all have an artifact (item 1, A10 fidelity, is documented in `assumptions.md` -- confirming with staff is the only open part) |
| 17 | Canonical-state analyzer: 2380 states -> equivalence classes -> geometric templates + stage matrix per class | `soccer_nash/templates.py` (`mirror_reduce` -> carrier-frame clustering), `docs/templates.md`: 94 -> 47 mirror pairs -> 8 templates, matrix per template |
| 18 | Characterize *why* multi-cell goal + interception geometry forces mixing | `templates.md` "Why the mix is unavoidable" (4-step argument + remove-any-ingredient); the analytic condition is a stochastic order **and** goal mouth > 1 cell **and** a defender that can contest but not cover both lanes |

## The measurement pass (4, 5, 6, 12)

- **4** -- `phase_diagram.csv` covers every board `3<=w<=11`, `3<=h<=9`,
  `w*h<=45` at every goal width. `--analyze`: goal width `1` -> **0 mixed states
  on every board**; goal width `>=2` -> mixing on every board. Among the boards
  that mix, an OLS of the mixed *fraction* is carried by board **area** (a
  dilution effect); goal width past 2 has a small negative coefficient. So goal
  width is a switch, not a dial, and board size only dilutes.
- **5** -- every neural fit and every stochastic rollout now takes `--seeds`
  and reports mean +/- sd: DQN (`nash_dqn_seeds.csv`), the A10 imitation net,
  the A10 competition nets (`a10_competition_seeds.csv`), and Nash-vs-Nash
  self-play. Exact dynamic-programming results carry no seed by construction.
- **6** -- `benchmark.py` times the hybrid over 5 repeats and reports the
  median, plus `LP calls`, `states needing LP` (3.95%, exact and portable),
  `LP/state/sweep`. The report separates this from the wall-clock ratio.
- **12** -- the exact-vs-DQN table is six measured rows over 5 seeds, with the
  ground-truth column filled in and a note that the network only handles the
  deterministic game.

## Still open (not part of this review)

A board-size-free proof of the single-cell theorem's carrier half
(`docs/proof.md`), and confirming the three interpreted A10 collision rules with
course staff (`docs/assumptions.md`).
