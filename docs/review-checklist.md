# Review checklist

Status of the 20 review points.

## Terminology & claims

- [x] 1. Exact A10 transition semantics -- `docs/assumptions.md` lists every
  interpreted collision sub-case and what to confirm with course staff.
- [x] 2. Claim A / B / C distinguished in the report, findings, README,
  `assumptions.md`.
- [x] 3. `mode="pure"` documented as the maximin *security* value, not a Nash
  equilibrium value (nash_q.py docstring, report).
- [x] 8. "3.9% for this configuration" language; goal-mouth sweep replaces the
  "big board" framing.
- [x] 18. "matching-pennies-like / 2x2 support" wording; strict-dominance
  reduction only collapses 4 of 94 -- stated explicitly.

## Experiments & code

- [x] 4/5. `soccer_nash/geometry.py` + `tree.py` + `scripts/geometry_model.py`
  -- decision tree predicts pure/mixed at precision 0.96 / recall 0.91.
- [x] 6/15. Mechanism: goal-mouth width is the gate (`scripts/experiments.py
  goalmouth`); template table by (support, defender geometry) in
  `docs/geometry.md`.
- [x] 7. `experiments/board_sweep.csv` (width 3-11 x height 3-9) +
  `goal_mouth_sweep.csv`.
- [x] 10. `soccer_nash/nash_dqn.py` + `scripts/nash_dqn.py` -- exact vs neural
  Nash-Q on value error, action agreement, exploitability, runtime.
- [x] 11. Self-play repositioned as "secondary validation" (report section 8).
- [x] 16. `experiments/gamma_sweep.csv` -- gamma 0.5 .. 0.995.
- [x] 17. `experiments/tolerance_sweep.csv` -- rel_tol 1e-12 .. 1e-1.
- [x] 19. `soccer_nash/experiment.py` (`run_config` -> flat row) +
  `scripts/experiments.py` -> `experiments/*.csv`.
- [x] 20. `soccer_nash/symmetry.py` + `NashQIteration.run_symmetric()` --
  1190 mirror pairs, `V(mirror(s)) = -V(s)`, verified equivariant.

## Structure

- [x] 12/13/24. Report restructured as Problem / Method / RQ1-4 / Mechanism /
  Limitations / Algorithm / Secondary validation / Open questions, with the
  four RQs.
- [x] `A10SoccerGame` (exact assignment) vs `SoccerGame` (flexible, research
  variants); a10 scripts use `A10SoccerGame`.
