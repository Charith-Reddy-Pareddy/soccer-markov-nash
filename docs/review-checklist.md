# Review checklist

Tracking the 20 points from the review. `[x]` done, `[~]` partial, `[ ]` open.

## Terminology & claims

- [ ] 1. Verify exact A10 transition semantics -- add an explicit assumptions
  doc listing the interpreted collision sub-cases; flag as "to confirm with
  course staff".
- [ ] 2. Precise language for the existence result -- distinguish Claim A
  (every converged stage game has a pure saddle), Claim B (stationary pure
  equilibrium of the Markov game), Claim C (the theoretical A10 game).
- [ ] 3. "pure Nash" vs "pure-saddle / pure-security" -- `mode="pure"` takes the
  maximin security value, which exists even when it is below the minimax and is
  *not* a Nash equilibrium there.
- [ ] 8. "3.9% under the specified 7x5 config", not a universal constant.
- [ ] 18. Soften "matching pennies" to "matching-pennies-like strategic
  structure", or prove strategic equivalence via the payoff inequalities.

## Experiments & code

- [ ] 4/5. Geometry -> equilibrium-type model: per-state spatial features,
  predict {pure, degenerate, mixed}.
- [ ] 6/15. Minimum geometric configuration for unavoidable mixing; reduce each
  mixed game to essential support, symmetry-normalise, cluster into templates.
- [ ] 7. Board-size heatmap: width 3..11, height 3..9, P(mixed).
- [ ] 10. Exact hybrid Nash-Q vs DQN approximation: value error, action
  agreement, exploitability, convergence, runtime.
- [ ] 11. Reposition self-play/exploitability as a secondary validation
  experiment.
- [ ] 16. gamma sweep {0.5 .. 0.995}: mixed fraction, runtime, V(s0), max
  bracket error.
- [ ] 17. tolerance sweep {1e-12 .. 1e-3}: #pure / #degenerate / #mixed,
  max |V_hybrid - V_LP|.
- [ ] 19. Single experiment driver -> machine-readable CSV/JSON under
  `experiments/`.
- [ ] 20. Symmetry reduction: canonical representatives, solve one, reconstruct
  the mirror -- computational optimisation + structural property.

## Structure

- [ ] 12/13/24. Restructure the report as a paper (Problem / Method / RQ1-4 /
  Results / Mechanism / Limitations / New Algorithm / Open Questions) with the
  four research questions RQ1 Existence, RQ2 Efficiency, RQ3 Mechanism, RQ4
  Numerical robustness.
- [ ] Separate exact-A10 semantics from research variants
  (`A10SoccerGame` vs the flexible `SoccerGame`).
