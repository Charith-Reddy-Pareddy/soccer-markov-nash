# Docs

This directory is the technical notebook: every proof, every experiment,
reproducible from the repo's own code. For the technical answer, start with
**[report.pdf](report.pdf)** (or [report.md](report.md) / [report.html](report.html)).

- [design.md](design.md) — the game as a configurable family: every axis, the
  reward, the three transition rules, and the reason each is there.
- [policy_gradient.md](policy_gradient.md) (also [policy_gradient.pdf](policy_gradient.pdf)) —
  REINFORCE, A2C and PPO against the exact solver, by self-play and by
  fictitious play; the group-note checklist is [group_note.md](group_note.md) and
  every assumption is in [solver_assumptions.md](solver_assumptions.md).
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
- [tournament_deepdive.md](tournament_deepdive.md) — the causal version of
  that claim: patching greedy's policy with the exact Nash mix at only the
  mixed states recovers 43% of its robustness gap to minimax, one exploited
  state shown matrix and all, and a gamma sweep (`make tournament-deepdive`).
- [showcase.md](showcase.md) — six mixed states worked by hand: weighted-arrow
  diagrams, the exact stage matrix, and why relative position (not distance to
  goal) is what forces a guess (`make showcase`).
- [positions.md](positions.md) (also [positions.pdf](positions.pdf)) —
  fourteen player-position cases, one per canonical equilibrium-support shape
  and per mixing mechanism studied in the project (move order, tackle,
  reward, slip), including the exact board position drawn at the meeting,
  next to the full `4x4` Q matrix -- rows and columns fixed as player 0 and
  player 1 for every state, never reoriented by who has the ball -- redrawn
  as a node-and-arrow best-response graph (`make positions`,
  `make positions-pdf`); every case also carries a rounding diagnostic (the
  same matrix solved again at 3/2/1-decimal precision -- 2 decimals usually
  preserves the exact mix, 1 decimal often manufactures a pure saddle that
  isn't really there); Case 3 additionally shows two of its cells expanded
  by hand as `sum(P * (r + gamma * V(next)))` against the real
  `game.transitions()` outcomes and the solver's own `V`, including one
  where a successor is the state itself -- plus the
  minimax/always-left/random/best-response policy matrix
  (`make positions-matrix`).
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
- [a10_cases.pdf](a10_cases.pdf) — eight edge cases on the A10-deterministic
  board: every stage game there is provably pure (this project's own headline
  result), but ties, weak dominance, and the collision rule still hide real
  structure a single 100% action doesn't show on its own (`make a10-cases`,
  `make a10-cases-pdf`); each case links straight to that position in
  [explorer.html](explorer.html). The other four board rules get the same
  case-study treatment, each documenting how many of its states genuinely
  need a mixed equilibrium (vs. exactly 0 of 2380 on A10):
  [coinflip_cases.pdf](coinflip_cases.pdf) (a fair coin decides contests
  instead of the carrier — still 0 mixed, surprisingly),
  [tackle_cases.pdf](tackle_cases.pdf) (a 50/50 duel, 56 of 760 mixed — the
  richest source), [territory_cases.pdf](territory_cases.pdf) (a dense
  position reward, 69 of 2380 mixed), and [slip_cases.pdf](slip_cases.pdf)
  (movement noise, 52 of 1200 mixed) — `make board-cases-pdfs` regenerates
  all five.
- [explorer.html](explorer.html) — interactive board: place both players
  anywhere on any of the six boards (the two canonical boards side by side,
  plus the three used in positions.md and the coin-flip contest-resolution
  variant) and get the live equilibrium policy and exact Q matrix (table or
  best-response graph) for all 11,480 states, plus that state's own rounding
  diagnostic (same
  3/2/1-decimal re-solve as positions.md and numerics.md, shown whenever
  rounding actually changes the classification or support — most states
  report that it doesn't). A React app (source in [../site/](../site/), data
  from `make explorer-data` / `scripts/explorer_data.py`, built with
  `make site` into this page and [index.html](index.html) — see
  `site/README.md`).
- [gallery.html](gallery.html) — every policy and value surface, drawn
  (`make gallery`, from `soccer_nash/viz.py`).
- [figures/](figures/) — SVGs from `make figures` (`soccer_nash/render.py`).

## The research questions

- [geometry.md](geometry.md) — RQ3: which spatial configurations force mixing,
  and the decision tree that predicts them.
- [templates.md](templates.md) — the 94 no-pure-saddle states reduced to 8 geometric
  templates, with the stage matrix and matching-pennies proof for each; plus
  whether the fifth action (`STAND`) changes the count (94→102, 8→9 --
  `make templates-n5`).
- [result.md](result.md) — the goal-width switch stated precisely: single cell →
  pure everywhere, goal ≥ 2 cells → a provably saddle-free stage game; the
  per-stage-game certificate, the robustness envelope, and the prior-work
  positioning (`make verify`).
- [generalize.md](generalize.md) — how far the switch holds: it survives board
  scale, aspect ratio, and goal placement -- including a goal spanning the
  entire boundary, which changes which states are mixed (Case 3 turns pure)
  without breaking the switch itself -- but action-independent movement
  noise (`slip`) does break it, so the switch is specific to Littman's
  move-order rule (`make generalize`).
- [mechanism.md](mechanism.md) — why a one-cell goal makes every stage game pure.
- [proof.md](proof.md) — the single-cell pure-saddle theorem: the defender's
  closed-form optimal strategy, the dominance-solvability certificate
  (`make proof`), and the gap that remains.
- `experiments/phase_diagram.csv` + `figures/png/phase_diagram.png` — the
  goal-width × board-size phase diagram (`make phase`).
- [numerics.md](numerics.md) — why the LP returns pure strategies, the value
  bracket, why 0.1-rounding is never safe (10 of the 13 genuinely mixed
  cases on [positions.md](positions.md) flip to a spurious pure saddle at
  1-decimal precision, checked case by case with `rounding_diagnostic`),
  and value-iteration vs. freeze-then-iterate — all with figures
  (`scripts/numerics.py --figures`).
- [degeneracy.md](degeneracy.md) — not every fractional LP output is a forced
  mixture: 64 of the 94 no-pure-saddle states are a unique forced mix, 30 have
  a zero-weight action tied with the reported support (`make degeneracy`).
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
- [neural.md](neural.md) — Neural Nash-Q vs. the exact solver (full tables;
  report section 8 has the summary): multi-seed, three Q-net starting points
  (from-zero, fit-to-exact, warm-start) plus a policy net, on both the
  deterministic game (`experiments/nash_dqn_seeds.csv`, `make dqn`) and the
  random-move-order game where equilibria are genuinely mixed
  (`experiments/nash_dqn_random_seeds.csv`, `make dqn-random`); a
  width/depth capacity sweep up to 512-wide / 115x the original architecture
  (`experiments/nash_dqn_ablation.csv`, `make dqn-ablation`).
- [policy_gradient.md](policy_gradient.md) — self-play REINFORCE vs. the
  exact solver: from a random init it does not reliably converge (33%
  action agreement, exploitability 0.86); pre-training onto the exact
  policy reaches 98% agreement, and 2000 further iterations of self-play
  leave it numerically unchanged -- a vanishing score-function gradient
  once the policy is peaked, not a fix; batching several independent
  rollouts per gradient step (instead of one correlated trajectory) cuts
  seed-to-seed variance to zero but by collapsing every seed onto the same
  state-independent policy, not by converging closer to equilibrium; and
  sharing weights between the two players' networks (fully, via one shared
  net reading the raw joint state, or via a shared trunk -- deliberately
  not via the game's own mirror symmetry, since a symmetric game is not
  guaranteed to have only symmetric equilibria) makes the result *more*
  exploitable on average than two fully independent nets, not less; giving
  batching a fair update budget instead (8x the data, not 8x fewer updates)
  stops the collapse and lands roughly even with the correlated baseline;
  and a learned baseline and an entropy bonus, tested separately, are not
  the same lever -- entropy sharply cuts exploitability's seed-to-seed
  variance without fixing convergence, the baseline does much less than
  either on its own (`make policy-gradient`, `make policy-gradient-warmstart`,
  `make policy-gradient-batch`, `make policy-gradient-batch-updates`,
  `make policy-gradient-architectures`, `make policy-gradient-ablation`).

