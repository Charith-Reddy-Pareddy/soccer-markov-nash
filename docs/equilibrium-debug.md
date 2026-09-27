# Equilibrium policy audit

The reported state `(4, 3, 5, 3, 0)` now selects **P0 Down, P1 Right**.

## Root cause

The original hybrid solver selected the first action with maximal worst-case
payoff. Up, Down, and Left have the same guarantee, zero, but they are not
interchangeable against every defender action. Down's row is
`[0.729, 0, 0.729, 0]`; Up and Left have all-zero rows. Down weakly dominates
both: never worse, strictly better against Up and Left. Right remains unsafe
because the swap against Left gives P0 `-0.59049`.

The explorer then replaced the solver's one-hot strategy with a uniform mix
across security ties. That mix is a Nash equilibrium of this matrix, but it
includes weakly dominated actions and is not a uniquely implied policy.
Equal worst-case values do not imply a mandatory equal-probability mixture.

The fix selects the best security guarantee first, and uses mean payoff
against opponent actions only to break **exact** security ties. It cannot
trade safety for average payoff. Both players use the same rule applied to
their own payoff matrix. If both criteria tie, action order gives a
reproducible representative; uniqueness is not claimed. Mixed games continue
to use the LP solver. The browser displays and samples the exported policy.

## Transition and value audit

All 16 manually calculated successors of Case 1 match the transition engine.
Against P1 Right, P0 Up/Down/Left/Right respectively reach:

| P0 | Successor | Immediate reward | Continuation value |
|---|---|---:|---:|
| Up | `(4,4,6,3,0)` | 0 | 0 |
| Down | `(4,2,6,3,0)` | 0 | 0 |
| Left | `(3,3,6,3,0)` | 0 | 0 |
| Right | `(5,3,6,3,0)` | 0 | 0 |

These values assume optimal future defense. They do not assume P1 continues
Right forever. No transition or reward was changed to manufacture Down.
The values are for the stationary discounted game, gamma 0.9; the finite
100-turn undiscounted assignment is a different objective.

The exported-data regression checks all 11,480 states across six boards:
probability normalization, equilibrium bounds, every transition and reward,
Bellman reconstruction, and non-overlapping nonterminal positions. Selected
pure actions are also checked against weak dominance.

## Other edge cases

| Case | Selected P0 | Selected P1 |
|---|---|---|
| 1: swap trap | Down | Right |
| 2: pinned corner | Right | Down |
| 3: standoff | Down | Up |
| 4: open goal | Right | Right |
| 5: open net | Up | Left |
| 6: getaway | Right | Down |
| 7: tightest tie | Right | Right |
| 8: dead zone | Up | Up |

Case 8 has an entirely zero matrix; the selection convention has no payoff
basis for preferring any direction. Case 4's defender cannot prevent the goal.
These limitations are stated instead of inventing geometric preferences.

## Additional defects repaired

- The best-response graph used independent first-occurrence tie breaks to
  find saddles and drew improvement arrows between equal-payoff actions.
  It now uses the security certificate and strict improvement arrows.
- Near-degenerate matrices reached by policy iteration can cause HiGHS to
  return an unknown status. A fallback uses its interior-point method with
  tighter tolerances and presolve disabled, on the original constraints.
  Failure is still reported if the fallback fails.
- Sampling at random draw zero could select a zero-probability action.
  Cumulative sampling now uses a strict comparison.
- Territory simulations omitted intermediate rewards. Exported rewards stay
  aligned with sampled outcomes; Step/Play/Back and simulations accumulate
  the same discounted rewards used by the solver.
- Six-decimal exports discarded computational precision. Values, policies,
  matrices, and transition probabilities now retain full floating precision;
  formatting is applied only when displayed.
- `make test-all` previously inherited pytest's exclusion of slow tests.
  It now clears the marker filter. CI also tests, lints, and builds the site.

## Reproduction

```sh
python -m pytest -m ""
python -m ruff check .
python scripts/explorer_data.py
python scripts/a10_cases.py
cd site
npm ci
npm test
npm run lint
npm run build
```

Baseline: 342 tests passed; 44 slow tests were excluded by the old default.
New policy regressions initially failed in three cases. All 16 hand-calculated
transitions passed. The two policy-iteration numerical regressions now pass.
See the task's final verification results for the complete suite outcome.
