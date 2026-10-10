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

### Speed-ups (none in Littman's paper)

Littman's minimax-Q solves an LP at every `(s, a, o)` update. This solver adds:

| trick | effect | where |
|---|---|---|
| **precompute the outcome table** -- build every stage game's transition list once in `__init__`, then look up `s'` during the recursion | "solve once for every `s`" -- turns per-entry transition work into a dict lookup | `NashQIteration.__init__` (`_out`) |
| **pure-first** -- the `O(A²)` best-reply check before any LP; use the saddle value when one exists | `~25×` fewer LP calls (3.95% vs 100% of states) | `_stage_value`, `matrix_games.pure_saddle_points` |
| **LP-result memoization** -- key the LP value on `round(M, 11).tobytes()` | near convergence most stage matrices repeat sweep-to-sweep, so `~10 000` LP calls become `~9 700` -- a further few percent | `_cached_game_value` (`_lp_cache`) |
| **mirror symmetry** -- solve one state per board-flip/player-swap pair, reconstruct the other by `V(mirror(s)) = -V(s)` | `2×` on the deterministic game (`0.33 s → 0.23 s`) | `run_symmetric`, `symmetry.py` |
| **`eval_value="mid"`** -- back up `p M q` (not `min(pM)` / `max(Mq)`) in the frozen policy-iteration sweep | `2×` fewer outer rounds than the bounds ([numerics.md](numerics.md) §3) | `run_policy_iteration` |

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
| templates | `templates.py` | the 94 no-pure-saddle states → mirror pairs → geometric templates | ~15 s |
| certificate | `onecell_proof.py` | single-cell proof, 13 boards × 5 discounts, 3 checks each | ~4 min |
| `benchmark.py` | — | hybrid solve, **median of 5 repeats**, LP-call rate | ~2 min |
| `nash_dqn_seeds.csv` | `nash_dqn.py --seeds 5` | neural Nash-Q vs exact -- from-zero, fit-to-exact, warm-start Q nets + policy net, **5 seeds** | ~16 min |
| `nash_dqn_ablation.csv` | `nash_dqn_ablation.py --seeds 2` | Q-net width (32-256) + depth (1-4 layers) sweep, fit-to-exact, **2 seeds/config x 10 configs** | ~17 min |
| `tournament_deepdive.csv` | `tournament_deepdive.py` | patched-policy causal test + gamma sweep (6 discounts) of the greedy-vs-minimax robustness gap | ~80 s |
| `nash_dqn_random_seeds.csv` | `nash_dqn_random.py --seeds 5` | neural Nash-Q on the RANDOM move-order game (mixed stage games, LP-exact bootstrap target) -- same 3 Q-net starting points + policy net, **5 seeds** | ~30 min |
| `degeneracy.csv` | `degeneracy.py` | classify all 94 no-pure-saddle states: unique forced mix vs. degenerate (zero-weight action tied with the reported support) | ~10 s |
| `distance_table.csv` | `distance_table.py` | no-pure-saddle / forced / degenerate counts, bucketed by Manhattan distance between the players | ~10 s |
| `pure_vs_mixed_exploit.csv` | `pure_vs_mixed_exploit.py` | greedy-pure vs. Nash-mixed value against a best response, at each positions.md case | instant |
| `a10_competition_seeds.csv` | `a10_competition.py --seeds 5` | the two competition nets, **5 seeds** | ~15 min |
| `pg_finite_a10.csv` | `pg_finite.py --seeds 3` | discounted 100-step game: REINFORCE / A2C / PPO by self-play and by fictitious play vs. the exact finite-horizon solution, **3 seeds** | ~40 min |
| `pg_finite_rate_a10_*.csv` | `pg_finite.py --scoring rate --algos <algo> --modes <mode> --tag _<algo>_<mode>` | the continuing game (play restarts after a goal, fixed 100 steps; a win is more goals), including A2C with the exact solver's values as a frozen critic (`a2c_exact`); **3 seeds** | ~30 min each |
| `pg_variant_{random,a10,rate_a10}_{shared,trim10}.csv` | `pg_finite.py --board <random\|a10> [--scoring rate] --algos a2c ppo --iterations 2000 --shared --tag _shared` (and `--trim 10 --tag _trim10`) | A2C and PPO with one shared network (two output heads), and with the last 10 steps of each episode left out of the loss, on all three boards; **3 seeds** | ~1 h each |
| `pg_variant_{random,a10}_argmax_{reinforce,a2c,ppo}.csv` | `pg_finite.py --board <random\|a10> --algos <algo> --modes fictitious_argmax --iterations 2000 --tag _argmax_<algo>` | fictitious play whose snapshots are pure (argmax) policies, for the opponent mixture and the reported policy; **3 seeds** | ~1-2 h each |
| `rps_nn.json` | `rps_nn.py` | rock-paper-scissors with neural-network players: standard, fictitious play, argmax fictitious play, at the soccer settings and with a larger entropy bonus; **3 seeds**, 3,000 iterations | ~15 min |
| `pg_random_kickoff_value.json` | `pg_kickoff_value.py` | value of the random board at the kickoff for the exact 100-step solution (the ball-holding seat's edge) | ~1 min |
| `pg_mixed_states.json` | `pg_mixed_states.py --algo <algo> --mode <mode>` for each of the six learners, then `--merge` | the learners at the random board's 94 mixed states: win rates from games started there, distance from the exact mix, equilibrium regret, three states in full; **3 seeds** | ~25 min each |
| `pg_finite_random_{reinforce,a2c,ppo}.csv` | `pg_finite.py --board random --algos <algo> --tag _<algo>` | the same six learners on the random move-order board (94 mixed stage games), discounted 100-step objective, **3 seeds** | ~30 min each |
| `pg_finite_a10_long_*.csv` | `pg_finite.py --iterations 8000 --modes selfplay --seeds 2` | the same self-play learners at 4x the training, **2 seeds** | ~15-35 min each |
| `pg_fp_br_*.csv` | `pg_fp_br.py --algo a2c` | fictitious play with best-response phases (100 or 300 policy-gradient iterations per best response), exploitability every 5 rounds | ~8 min |
| `site/src/pgResults.json` | `pg_site_data.py` | every number on the site's Policy gradient tab, regenerated from the result CSVs (a test checks the file against them) | instant |
| `pg_policy_outputs.json` | `pg_policy_outputs.py --algo a2c --mode selfplay`, then `--merge` | action probabilities (U, D, L, R) of the six learners at four fixed positions, shown on the site and in the PDF | ~3 min each |
| `dqn_finite_a10.csv` | `dqn_finite.py --seeds 3` | Nash-DQN on the discounted 100-step game, same scoring, **3 seeds** | ~10 min |
| `pg_algos_{a10,random}_seeds.csv` | `pg_algos_and_fp.py pg --board ... --seeds 3` | self-play REINFORCE vs. A2C vs. PPO vs. the exact solve, + win/tie/loss and mirror gap, **3 seeds** | ~10 min each |
| `pg_algos_random_explore_seeds.csv` | `pg_algos_and_fp.py pg --board random --explore --seeds 3` | same with exploring starts | ~10 min |
| `dqn_winrates.csv` | `pg_algos_and_fp.py dqn --seeds 3` | Nash-DQN win/tie/loss vs. random, exact Nash and best response | ~4 min |
| `markov_fictitious_play.csv` | `pg_algos_and_fp.py markov-fp` | fictitious play inside the Markov game, restart vs. persistent beliefs | ~1 min |
| `fictitious_play.csv` | `pg_algos_and_fp.py fp` | best-response dynamics vs. fictitious play on RPS and on all 2,380 stage games | ~5 s |
| self-play | `selfplay.py --seeds 5` | Nash-vs-Nash return, **5 seeds × 3000 games** | ~5 min |
| `board_sweep.csv` | `experiments.py board` | larger board sweep (legacy) | ~10 min |

`pg_finite.py` trains for 2,000 iterations of 64 episodes unless `--iterations` is given (the default, so `pg_finite.py --seeds 3` is the 2,000-iteration run); new output files record the count in an `iterations` column.

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

**Tests.** 484 fast tests (`pytest -m "not slow"`) pass on every commit --
collision-rule enumeration against the A10 spec, the self-loop `V = 0`
invariant across all 2380 states, solver/backup equivalence, and the neural
(DQN/policy-gradient) and reward-audit modules. `make test-all` adds ~34
slower, seed-dependent experiment reproductions.

## Reproducing

```bash
python -m venv --system-site-packages .venv   # see README for the macOS SciPy note
source .venv/bin/activate
pip install -r requirements.txt -e .
make lint            # ruff
make test-all        # ~315 fast + slow tests
make experiments     # regenerates the CSVs above
make phase templates proof benchmark dqn figures report
```

Environment: Python ≥ 3.10, `numpy`, `scipy` (HiGHS), `ruff`, `coverage`,
`pytest`. No GPU. `docs/report.pdf` is rendered from `docs/report.html` with
headless Chrome.
