# Docs

Start with **[report.pdf](report.pdf)** (or [report.md](report.md) /
[report.html](report.html)) — the full answer to the research question.

## Background and scope

- [assumptions.md](assumptions.md) — the environment, the three claims, and
  which A10 collision rules are quoted vs. interpreted.
- [questions-for-staff.md](questions-for-staff.md) — the specific things to
  confirm with course staff (redacted parameters, the unspecified collision
  sub-cases, the discount).

## The research questions

- [geometry.md](geometry.md) — RQ3: which spatial configurations force mixing,
  and the decision tree that predicts them.
- [templates.md](templates.md) — the 94 mixed states reduced to 8 geometric
  templates, with the stage matrix and matching-pennies proof for each.
- [mechanism.md](mechanism.md) — why a one-cell goal makes every stage game pure.
- [proof.md](proof.md) — the single-cell pure-saddle theorem: the defender's
  closed-form optimal strategy, the dominance-solvability certificate
  (`make proof`), and the gap that remains.
- `experiments/phase_diagram.csv` + `figures/phase_diagram.svg` — the
  goal-width × board-size phase diagram (`make phase`).
- [numerics.md](numerics.md) — the value bracket, stage-game classification, and
  how rounding under discounting misfires.
- [findings.md](findings.md) — dated working notes (Q1–Q7).

## A10 deliverables

- [a10_part1.md](a10_part1.md) — successor states and transition rewards.
- [a10_part2.md](a10_part2.md) — best response to the scripted opponent, imitated
  by a bias-free policy network.
- [a10_competition.md](a10_competition.md) — the two equilibrium networks and how
  exploitable the approximation is.

## Extensions

- [selfplay.md](selfplay.md) — Nash-vs-Nash return and exploitability.
- [shaping.md](shaping.md) — potential-based reward shaping leaves the
  equilibrium fixed; a naive possession bonus does not.
- Neural Nash-Q vs. the exact solver — report section 10, multi-seed
  (`experiments/nash_dqn_seeds.csv`, `make dqn`).

## Provenance

- [review-checklist.md](review-checklist.md) — the first 20 review points.
- [review-18.md](review-18.md) — the second review (18 points) and its status.
- [figures/](figures/) — SVGs from `make figures` (`soccer_nash/render.py`).
