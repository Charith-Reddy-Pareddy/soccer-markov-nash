# Docs

Start with **[report.pdf](report.pdf)** (or [report.md](report.md) /
[report.html](report.html)) — the full answer to the research question.

- [design.md](design.md) — the game as a configurable family: every axis, the
  reward, the three transition rules, and the reason each is there.
- [methods.md](methods.md) — the solver, the state encoding, seeds, repeat
  counts, board ranges: how every number was produced.
- [discussion.md](discussion.md) — research-level open questions, the best
  answers the evidence supports, and what the mixed region depends on.
- [advisor.md](advisor.md) — how the research-meeting feedback maps to the repo.
- [littman.md](littman.md) — Littman 1994, the fifth action (`stand`), and a
  reproduction of his Figure 2 mixed equilibrium (`make littman`).
- [tournament.md](tournament.md) — Littman's Table 3, reproduced exactly:
  minimax exploits *and* survives its challenger; greedy policies do one or the
  other, never both (`make tournament`).
- [mixing.md](mixing.md) — how shallow the mixed stage games are, their
  matching-pennies structure, and the value of mixing (`make mixing`).
- [reward.md](reward.md) — a second reward objective (`scoring="rate"`: goal
  reset, expected goal difference); the mixed region is invariant to it
  (`make reward`).
- [blend.md](blend.md) — sweeping the move-resolution rule from deterministic to
  random; mixing switches on sharply above `blend = 0.5` (`make blend`).
- [occupancy.md](occupancy.md) — the mixed states carry ~40% of the equilibrium
  path despite being 4% of the state space (`make occupancy`).

## Background and scope

- [assumptions.md](assumptions.md) — the A10 geometry instance and which of its
  collision rules are quoted vs. interpreted.
- [gallery.html](gallery.html) — every policy and value surface, drawn
  (`make gallery`, from `soccer_nash/viz.py`).

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
- [numerics.md](numerics.md) — why the LP returns pure strategies, the value
  bracket, why 0.1-rounding is never safe, and value-iteration vs.
  freeze-then-iterate — all with figures (`scripts/numerics.py --figures`).
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
