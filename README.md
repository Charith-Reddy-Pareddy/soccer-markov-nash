# When Does a Soccer Markov Game Need Mixed Strategies?

A pure-first Nash-Q approach. Research code built on the CS 540 A10 soccer
assignment (<https://pages.cs.wisc.edu/~yw/CS540S26A10.html>).

## Research question

> Which geometric configurations of the soccer Markov game make mixed strategies
> necessary, can those configurations be characterized analytically, and can a
> solver that checks for a pure equilibrium first exploit that structure?

**Full answer: [docs/report.pdf](docs/report.pdf)**, also as
[docs/report.html](docs/report.html) (open in any browser) and
[docs/report.md](docs/report.md).

Two contributions:

- **Algorithmic** &mdash; a pure-first hybrid Nash-Q backup: check each stage
  game for a pure saddle, take its value when it exists, fall back to the LP
  only where it does not. Exact, and it skips the LP on 96&ndash;100% of states.
- **Structural** &mdash; a characterization of *when* a stage game is
  intrinsically mixed. For the implemented deterministic A10 model every
  converged stage game has a pure saddle. Mixing appears only under Littman's
  random move *order*, only when the goal mouth is wider than one cell, and only
  where the defender can contest the carrier's forward cell but not cover every
  scoring lane in one move (3.95% of states on the 7x5 3-cell-goal board). Those states reduce
  to a few matching-pennies templates.

- Scope, the three claims, interpreted collision rules: [docs/assumptions.md](docs/assumptions.md)
- How every number was produced: [docs/methods.md](docs/methods.md)
- Research-level open questions and their best answers: [docs/discussion.md](docs/discussion.md)
- Where mixing is forced, and why: [docs/geometry.md](docs/geometry.md), [docs/templates.md](docs/templates.md), [docs/mechanism.md](docs/mechanism.md)
- Everything else: [docs/README.md](docs/README.md)
- `make test` / `make experiments` / `make report` regenerate everything

## Results at a glance

The A10 game is undiscounted; its exact solution is 100-step backward induction
(`experiments/undiscounted.csv`).

| move order | mixed stage games (state × step) | V(kickoff) | pure eq? |
|---|---|---|---|
| `deterministic` (A10) | **0** of 238 000 | 0.000 | yes (non-stationary) |
| `coinflip` tie-break | **0** | 0.000 | yes |
| `random` (Littman) | 3 148 (404 states) | +0.459 | no |

The stationary `gamma < 1` solve agrees and is faster: 0 no-saddle stage games
on the deterministic game at every `gamma` in 0.5–0.995; `94 / 2380` on the
random game at `gamma = 0.9`. `V(kickoff)` is at the A10 netID start `(0,1,6,3,0)`;
the mixed count does not depend on it.

The pure-first hybrid solver reproduces the all-LP value function to `4e-16`
while calling the LP on only 3.95% of states (~25x fewer per sweep); mirror
symmetry halves the remaining work. LP-solve reduction and wall-clock speedup
are reported separately &mdash; the former is a property of the game, the latter
depends on the machine and LP backend.

## Repository map

| path | what |
|---|---|
| `soccer_nash/game.py` | the Markov game (`SoccerGame`, `A10SoccerGame`) |
| `soccer_nash/matrix_games.py` | zero-sum stage solvers: pure saddle + LP minimax |
| `soccer_nash/nash_q.py` | Nash Q-iteration (`pure` / `mixed` / `hybrid`, VI and PI) |
| `soccer_nash/support_enum.py`, `markov_game.py` | all equilibria / general-sum extension |
| `soccer_nash/numerics.py` | value bracket, stage-game classification, rounding |
| `soccer_nash/geometry.py`, `tree.py` | state features and the mixed-state decision tree |
| `soccer_nash/symmetry.py` | mirror states, symmetric solve |
| `soccer_nash/render.py` | draw a state as SVG (player 0 blue, player 1 green) |
| `soccer_nash/a10.py`, `mlp.py`, `opponents.py`, `best_response.py` | A10 deliverables |
| `soccer_nash/shaping.py`, `exploit.py`, `nash_dqn.py` | reward shaping, self-play, neural Nash-Q |
| `scripts/` | one entry point per analysis; all write to `docs/` or `results/` |
| `experiments/*.csv` | committed sweep outputs the report tables read from |

## Plan

1. **Environment** — reproduce the A10 two-player soccer Markov game on a 7x5
   grid. Three move-resolution rules: `deterministic` (A10), `random` (Littman's
   random move order), `coinflip` (a fair-coin tie-break).
2. **Stage-game solvers** — pure saddle-point finder and an LP solver for
   zero-sum mixed Nash.
3. **Nash Q-iteration** — three stage solvers (`pure` / `mixed` / `hybrid`),
   run by either value iteration (`run()`) or freeze-then-iterate policy
   iteration (`run_policy_iteration()`).
4. **Analysis** — where the value functions and policies agree / diverge,
   self-play, exploitability, reward shaping.

## Headline result

- **Deterministic game (exact A10):** solved exactly, undiscounted, by
  backward induction -- every one of the 238 000 (state × step) stage games has
  a pure saddle. A pure equilibrium exists; the discount was never load-bearing.
- **Random move order (Littman):** 3.95% of stage games on the 7x5 3-cell-goal
  board have no pure saddle, so no pure stationary equilibrium exists; the
  `pure` solver then under-values the kickoff by 0.16. **Whether any state needs
  mixing is decided entirely by goal-mouth width** -- one cell: never; two or
  more: always ([docs/geometry.md](docs/geometry.md), phase diagram).
- **Coinflip tie-break:** stochastic, yet its value function is identical to the
  deterministic game's -- it is Littman's move *order*, not randomness, that
  forces mixed strategies.

See [docs/findings.md](docs/findings.md) and [docs/README.md](docs/README.md).

## The pieces

| script | what | doc |
|---|---|---|
| `a10_part1.py` / `a10_part2.py` / `a10_competition.py` | the A10 deliverables (successor tables; imitation network; competition networks). `--seeds N` on the last two reports fit accuracy + exploitability over N seeds. | `docs/a10_*.md` |
| `selfplay.py --seeds 5` | Nash-vs-Nash return and exploitability | [docs/selfplay.md](docs/selfplay.md) |
| `policy_iteration.py` | value iteration vs. freeze-then-iterate: same fixed point, ~5x fewer LP solves | -- |
| `numerics.py` | value bracket, five-way stage-game classification, rounding | [docs/numerics.md](docs/numerics.md) |
| `templates.py` | the 94 mixed states -> 8 geometric templates | [docs/templates.md](docs/templates.md) |
| `phase_diagram.py` | goal-width x board-size phase diagram (`--analyze` decomposes the variance) | -- |
| `onecell_proof.py` | the single-cell pure-saddle certificate | [docs/proof.md](docs/proof.md) |
| `benchmark.py` | repeated-run timing + LP-call rate of the hybrid | -- |
| `nash_dqn.py --seeds 5` | neural Nash-Q vs. the exact solver | -- |

`soccer_nash.shaping` adds intermediate rewards: potential-based shaping leaves
the equilibrium unchanged (value shifts by exactly `-Phi`), a naive possession
bonus changes it ([docs/shaping.md](docs/shaping.md)).

## Environment assumptions

The A10 page parameterises start positions and goal rows by student ID (redacted
in the public page). This repo uses the standard Littman soccer geometry:

- Grid: width 7 (`x` in `0..6`), height 5 (`y` in `0..4`).
- Player 0 starts left and scores through the right edge (`x = 7`); player 1
  starts right and scores through the left edge (`x = -1`).
- Goal mouth: middle rows `y in {1, 2, 3}`.
- Actions: `U, D, L, R` (no stay). Vertical moves clamp at the top/bottom walls;
  horizontal moves clamp except a ball carrier moving into the opponent goal.
- Collisions (`deterministic`): if both players target the same square, the ball
  carrier takes it and possession flips to the other player; on a swap, players
  swap and possession flips. A carrier blocked by a stationary opponent stays
  put and loses the ball.
- Collisions (`random`): the two moves are applied in a random order (each with
  probability 1/2); a move into the other player's current cell fails, and
  transfers the ball if the mover held it.
- Collisions (`coinflip`): the A10 rule, but a fair coin (not possession)
  decides who wins a contested square or a swap.
- Rewards: `+1` win, `-1` loss, `0` otherwise (zero-sum). Game ties after 100
  steps.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
make lint   # ruff check
make test   # fast suite; `make test-all` adds the slow solver runs
```

CI (`.github/workflows/tests.yml`) runs `ruff check` and the full `pytest` on
every push and pull request. `make coverage` reports line coverage (99%, with
the gaps in degenerate-case guards).

On very recent macOS builds the PyPI SciPy wheel can fail to load
(`_spropack.so` dyld error). If so, create the venv against a working
interpreter instead:

```bash
python -m venv --system-site-packages .venv
```

## License

MIT &mdash; see [LICENSE](LICENSE).
