# Third review -- 15 points

The DQN-fix / report-framing review. Each row names the concrete artifact
that closes the point. A follow-up 9-point list restated 8 of these 15
verbatim (items 1, 2, 3+7, 4, 5+6, 8, 9, 12+13 below cover all 9); the one
genuinely new item from that list was this page itself.

| # | point | where it lives |
|---|---|---|
| 1 | Fix the DQN story: run neural Nash-Q on the random-order/mixed game, not only deterministic | `soccer_nash/nash_dqn.py`'s `_minimax_batch` (pure-fast-path + LP fallback, not maximin-only) and `_precompute_transitions` (`game.transitions`, the real distribution, not `game.step`'s single sample); `scripts/nash_dqn_random.py` reruns from-zero/fit-to-exact/warm-start/policy-net on the random-move-order board (56/760 no-pure-saddle states); `experiments/nash_dqn_random_seeds.csv`; write-up in `report.md` §8 "The harder game" |
| 2 | A proper capacity sweep before concluding representation failure -- don't limit network size | `scripts/nash_dqn_ablation.py` extended to width 512 and a depth-3/width-512 point (536.6k params, 115x the original architecture); `experiments/nash_dqn_ablation.csv`; report §8 "Width/depth ablation" -- action agreement's best point (48.1%, 137k params) is not beaten at 537k params (47.5%), while exploitability keeps falling the whole way (0.79 &rarr; 0.05) |
| 3 | Separate no-pure-saddle states from fractional-but-degenerate LP policies; rename "94 mixed states" where that's what the certificate actually proves | `scripts/degeneracy.py` classifies all 94: 64 unique forced mix, 16 degenerate face, 14 pure-tied (always the defender); `docs/degeneracy.md`; "94 mixed states" &rarr; "94 no-pure-saddle states" across ~15 files wherever it's a count claim |
| 4 | Reconcile "every mixed state has a matching-pennies core" with the degeneracy finding | `docs/result.md` §3 and `docs/degeneracy.md`: the core is a property of the matrix (always present, all 94); degeneracy is a separate, LP-vertex-specific fact about which equilibrium got printed (30 of 94) |
| 5 | Distance-between-players analysis | `scripts/distance_table.py`: no-pure-saddle / forced / degenerate counts by Manhattan distance; confirms `showcase.md`'s existing hand-quoted 40-at-1/54-at-2/0-beyond figure exactly, and adds that distance 1 has a *higher* no-pure-saddle rate (17.2%) than distance 2 (14.4%) |
| 6 | Make relative-geometry / template collapse an explicit headline result, with a figure | `docs/templates.md`: boxed "94 no-pure-saddle states reduce to 8 canonical relative configurations under mirror/role-swap symmetry" statement; `scripts/templates.py` now also writes `docs/figures/gallery/templates.svg` (board + Q-matrix panel per template, 4 across) |
| 7 | Direct pure/greedy vs. Nash-mixed exploitability table at the same representative states | `scripts/pure_vs_mixed_exploit.py`: for 7 of `positions.md`'s cases, force the carrier to its most-likely action and let the defender best-respond within that stage game, vs. the actual Nash value; table in `positions.md`/`.pdf`, "value of mixing" never negative, 0.0011-0.0391 |
| 8 | Push the best-response-vs-challenger result more prominently -- move it earlier, answer "why does this matter" | new "Reading guide: core, supporting, exploratory" section right after the intro (previously this taxonomy only existed in §11 near the end); tournament result reclassified from "supporting" to "core"; §7 retitled "Why it matters: minimax survives what every deterministic policy cannot"; a "so what?" callout right after RQ3 points to it |
| 9 | Tighten README wording on the goal-width result -- "tested Littman family," not a universal law | `README.md`: "Within the tested Littman random-move-order family, one-cell goals produced no no-pure-saddle states..."; cross-references that `slip`/`tackle` break the switch even on a one-cell goal |
| 10 | Matrix-error / regret / support-agreement metrics, not just action agreement | `soccer_nash/nash_dqn.py`'s `compare_to_exact` gained mean/max Frobenius error, mean/max equilibrium regret (against the *exact* matrix, not the net's own), and row/col support agreement (new optional `exact_matrix_of`/`exact_col_policy` params, backward compatible) |
| 11 | "Action accuracy is not equilibrium accuracy" as a headline phrase | literal blockquote/note in `report.md`/`.html` right after the policy-net paragraph, and in the abstract's point 4 |
| 12 | Keep continuous-action work explicitly future work, not something already solved | new "Roadmap: this is not the eventual challenge" subsection: exact discrete &rarr; neural reproduction &rarr; continuous-action solver; explicit that no half-finished continuous solver belongs in the repo yet |
| 13 | Public-facing page: add the occupancy result and one DQN lesson | `the-mixed-game`: a third `StatTile` (41% occupancy on 4% of states) added to the existing grid; a new paragraph in "Why it matters" translating the action-accuracy finding to plain language |
| 14 | A status/checklist page for this review | this page |
| 15 | Everything else in the 15-point review not already covered above (terminology consistency, cross-links between the new docs) | `docs/degeneracy.md`, `docs/showcase.md`, `docs/templates.md`, `docs/result.md`, `docs/numerics.md`, `docs/findings.md` all cross-link each other and `docs/tournament_deepdive.md` |

## Verification

`ruff check .` clean; the full non-slow `pytest` suite green, re-run after
every major edit (`tests/test_degeneracy.py`, `tests/test_distance_table.py`,
`tests/test_pure_vs_mixed_exploit.py`, and new cases in `tests/test_nash_dqn.py`
for `_minimax_batch`, `_precompute_transitions`, and the stochastic-game
training path -- the old `test_rejects_stochastic_game` was removed since
rejecting stochastic games was exactly the bug being fixed). Both
`docs/report.pdf` and `docs/positions.pdf` regenerated and visually spot-checked
page by page. Both repos' GitHub Pages builds confirmed `built` and the live
PDFs / JS bundle spot-checked with `curl`/`grep` after every push.

## Still open (not part of this review)

Nothing from either the 15-point review or the follow-up 9-point list.
Longer-term, explicitly deprioritized: the continuous-action dog/soccer game
(§8's roadmap); a board-size-free proof of the single-cell theorem's carrier
half (`docs/proof.md`).
