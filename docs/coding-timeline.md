# Proposed ten-workday coding schedule

This is a planning estimate for an 8–10-hour working day, not a claim that
these hours have already been worked. Commits record actual work and dates.
Use short `debug ...` messages; no day labels or co-author trailers.

| Workday | Deliverable | Five reviewable checkpoints |
|---|---|---|
| 1 | Reproduce reported policy defect | baseline, transition fixtures, matrix regression, tie fix, focused verification |
| 2 | Audit all eight A10 edge cases | cases 1–2, cases 3–4, cases 5–6, cases 7–8, report consistency |
| 3 | Complete transition invariants | collisions, swaps, goals, wall clamps, possession symmetry |
| 4 | Certify solver behavior | pure ties, mixed policies, numerical failures, Bellman residuals, exploitability |
| 5 | Align explorer behavior | policy source, graph, sampling, step/back/restart, browser regressions |
| 6 | Verify reward objectives | goal rewards, territory rewards, stochastic outcomes, discounting, return comparisons |
| 7 | Validate exports | precision, normalization, transitions, state coverage, deterministic regeneration |
| 8 | Refresh research outputs | edge-case text, SVGs, PNGs, PDF, visual review |
| 9 | Run comprehensive validation | Python suite, frontend suite, lint, production build, independent diff review |
| 10 | Prepare handoff | remaining defects, reproduction instructions, limitations, final verification, review branch |

Checkpoint commits should contain meaningful work; do not create empty or
artificial commits merely to meet a count. Several checkpoints have already
been completed in this debugging session and do not need to be repeated.
Future work is not automatically scheduled. Stop at the requested 75% usage
threshold; available account counters are five-hour and weekly, not daily.
