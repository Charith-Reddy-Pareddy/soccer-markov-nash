# Docs

Non-technical, visual walkthrough: **[The Mixed Game](https://charith-reddy-pareddy.github.io/the-mixed-game/)**
([the-mixed-game](https://github.com/Charith-Reddy-Pareddy/the-mixed-game), a
separate repo). For the technical answer, start with **[report.pdf](report.pdf)**
(or [report.md](report.md) / [report.html](report.html)).

- [design.md](design.md) — the game as a configurable family: every axis, the
  reward, the three transition rules, and the reason each is there.
- [methods.md](methods.md) — the solver, the state encoding, seeds, repeat
  counts, board ranges: how every number was produced.
- [discussion.md](discussion.md) — research-level open questions, the best
  answers the evidence supports, and what the mixed region depends on.
- [littman.md](littman.md) — Littman 1994, the fifth action (`stand`), and a
  reproduction of his Figure 2 mixed equilibrium (`make littman`).
- [tournament.md](tournament.md) — Littman's Table 3, reproduced exactly:
  minimax exploits *and* survives its challenger; greedy policies do one or the
  other, never both (`make tournament`); the same result again with plain 4×4
  stage games and no `stand` action (`make tournament4`).
- [showcase.md](showcase.md) — six mixed states worked by hand: weighted-arrow
  diagrams, the exact stage matrix, and why relative position (not distance to
  goal) is what forces a guess (`make showcase`).
- [positions.md](positions.md) — player positions next to the stage game
  redrawn as a node-and-arrow best-response graph, the format sketched at the
  research meeting: one lit-up cell for a pure state, a closed loop of arrows
  for a mixed one (`make positions`).
- [mixing.md](mixing.md) — how shallow / how mixed the mixed stage games are
  (entropy: median 0.55 bits, 26/94 strongly mixed), their matching-pennies
  structure, and the value of mixing (`make mixing`).
- [reward.md](reward.md) — three reward objectives: `win`, `rate` (goal reset),
  and `territory` (dense final-third reward). `win` ↔ `rate` leave the mixed
  region identical; `territory` moves it, and gives the deterministic game
  matching-pennies stage games (`make reward`).
- [blend.md](blend.md) — sweeping the move-resolution rule from deterministic to
  random; mixing switches on sharply above `blend = 0.5` (`make blend`).
- [tackle.md](tackle.md) — the project's own `tackle` collision rule (a
  commit-to-the-challenge duel) and the rule-fingerprint panel comparing every
  collision rule's mixed region (`make tackle`).
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
- [result.md](result.md) — the goal-width switch stated precisely: single cell →
  pure everywhere, goal ≥ 2 cells → a provably saddle-free stage game; the
  per-stage-game certificate, the robustness envelope, and the prior-work
  positioning (`make verify`).
- [generalize.md](generalize.md) — how far the switch holds: it survives board
  scale, aspect ratio, and goal placement, but action-independent movement noise
  (`slip`) breaks it, so the switch is specific to Littman's move-order rule
  (`make generalize`).
- [mechanism.md](mechanism.md) — why a one-cell goal makes every stage game pure.
- [proof.md](proof.md) — the single-cell pure-saddle theorem: the defender's
  closed-form optimal strategy, the dominance-solvability certificate
  (`make proof`), and the gap that remains.
- `experiments/phase_diagram.csv` + `figures/png/phase_diagram.png` — the
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
