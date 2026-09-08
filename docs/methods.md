# Methods and reproducibility

How every number in the report was produced.

## The game

`soccer_nash/game.py`. State `(x0, y0, x1, y1, b)`: player 0 at `(x0, y0)`,
player 1 at `(x1, y1)`, `b ∈ {0, 1}` the ball carrier. Players never share a
cell, so a `w × h` board has `w·h·(w·h − 1)·2` non-terminal states (2380 on
7×5). Terminal states are `(-1, -1, -1, -1, winner)`. Actions `U D L R` map to
`(0, +1) (0, -1) (-1, 0) (+1, 0)` -- **`y` increases upward**. Rewards are `+1`
to player 0 on a player-0 goal, `-1` on a player-1 goal, `0` otherwise;
zero-sum; **undiscounted**; a tie (reward 0) after `max_steps = 100`.

Three move-resolution rules:

- `deterministic` -- the A10 rule: the ball carrier wins any contested square,
  a swap flips possession, a carrier that moves into a standing opponent stays
  and loses the ball. Transitions are a pure function of the joint action.
- `random` -- Littman's rule: the two moves are applied in a uniformly random
  order (each ordering probability `½`); a move into the other player's
  *current* cell fails and transfers the ball if the mover held it.
- `coinflip` -- the A10 rule but a fair coin (not possession) decides contests
  and swaps. `deterministic` is this with the carrier always winning.

`A10SoccerGame` is the exact assignment environment (deterministic only, and the
7×5 kickoff is this netID's, `(0,1,6,3,0)` -- pass `p0_start` / `p1_start` for a
different one). The interpreted collision sub-cases are in
[assumptions.md](assumptions.md).

## The solver

`soccer_nash/nash_q.py`, `soccer_nash/matrix_games.py`.

The game is zero-sum, so each state's stage game is a `4×4` payoff matrix
`M(s)[a0, a1] = E[R + γ·V(s')]` and the value is its minimax
(Shapley 1953). Three ways to solve it:

1. **`run_finite_horizon()` -- exact, undiscounted.** Backward induction:
   `V = 0` at the horizon, `V_t(s) = val(M_t(s))` for `t = 100 … 1`, `γ = 1`.
   Non-stationary. This is the actual A10 game.
2. **`run()` -- stationary value iteration, `γ < 1`.** A faster contraction that
   converges to a stationary fixed point; RQ4 shows the pure/mixed split is the
   same for every `γ` in `0.5 … 0.995`. `γ = 0.9`, `tol = 1e-9` unless noted.
3. **`run_policy_iteration()` -- freeze then iterate.** Freeze the stage
   strategies, run cheap linear evaluation sweeps, re-solve. Same fixed point,
   fewer LP solves (see [discussion.md](discussion.md) §4).

Each stage matrix is solved by one of three **backups**:

- **pure** -- the maximin security value `max_i min_j M[i,j]` (`O(A²)`, no LP). A
  lower bound; not the Nash value when it is below the minimax.
- **mixed** -- the LP minimax value (`scipy.optimize.linprog`, HiGHS).
- **hybrid** -- the pure saddle value where `maximin == minimax` (then it *is*
  the Nash value), the LP otherwise. Exact everywhere. LP results are memoized
  on the rounded matrix bytes, so a converging value function turns thousands of
  LP calls into hundreds.

A **stage game classification** (`classify_stage_game`, `numerics.py`) sorts
each stage matrix into five cases -- exact / numerically-indistinguishable /
strict / degenerate saddle, or genuine mixed -- with a tolerance that scales
with the matrix entries. Only "genuine mixed" forces the LP.

## The experiments

All `soccer_nash/experiment.py` and `scripts/*.py`; every table regenerates via
`make`.

| output | script | what it sweeps | cost |
|---|---|---|---|
| `undiscounted.csv` | `experiments.py undiscounted` | 100-step backward induction, 3 move orders | ~15 s |
| `baseline.csv` | `experiments.py baseline` | 3 move orders, `γ = 0.9`, + exploitability | ~30 s |
| `gamma_sweep.csv` | `experiments.py gamma` | `deterministic` + `random`, 8 discounts | ~2 min |
| `tolerance_sweep.csv` | `experiments.py tolerance` | `random`, 9 classification tolerances | ~2 min |
| `phase_diagram.csv` + heatmap | `phase_diagram.py` | every board `w ≤ 11, h ≤ 9, w·h ≤ 45` × every goal width, `random` | ~15 min |
| `phase_diagram.py --analyze` | — | OLS variance decomposition of the mixed fraction | instant |
| templates | `templates.py` | the 94 mixed states → mirror pairs → geometric templates | ~15 s |
| certificate | `onecell_proof.py` | single-cell proof, 13 boards × 5 discounts, 3 checks each | ~4 min |
| `benchmark.py` | — | hybrid solve, **median of 5 repeats**, LP-call rate | ~2 min |
| `nash_dqn_seeds.csv` | `nash_dqn.py --seeds 5` | neural Nash-Q vs exact, **5 seeds** | ~4 min |
| `a10_competition_seeds.csv` | `a10_competition.py --seeds 5` | the two competition nets, **5 seeds** | ~15 min |
| self-play | `selfplay.py --seeds 5` | Nash-vs-Nash return, **5 seeds × 3000 games** | ~5 min |
| `board_sweep.csv` | `experiments.py board` | larger board sweep (legacy) | ~10 min |

**Seeds.** Every measurement that depends on randomness -- neural-network
fitting, self-play rollouts -- is run over seeds `0 … 4` and reported as
`mean ± sd`. Exact dynamic-programming results (value iteration, backward
induction, LP solves, mixed-state counts) carry no seed.

**Runtime.** Wall-clock is a **median of 5 repeats** and is labelled
machine-dependent; the LP-call rate (`states needing LP`, `LP/state/sweep`) is
exact and portable, and is the headline efficiency number.

**Kickoff.** `V(kickoff)`, self-play returns and exploitability are measured at
the A10 netID start `(0,1,6,3,0)`. The pure/mixed counts and the whole geometric
analysis are over *all* states and do not depend on it.

## Reproducing

```bash
python -m venv --system-site-packages .venv   # see README for the macOS SciPy note
source .venv/bin/activate
pip install -r requirements.txt -e .
make lint            # ruff
make test-all        # ~200 fast + slow tests
make experiments     # regenerates the CSVs above
make phase templates proof benchmark dqn figures report
```

Environment: Python ≥ 3.10, `numpy`, `scipy` (HiGHS), `ruff`, `coverage`,
`pytest`. No GPU. `docs/report.pdf` is rendered from `docs/report.html` with
headless Chrome.
