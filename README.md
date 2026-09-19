# When Does a Soccer Markov Game Need Mixed Strategies?

This repo is the technical notebook: every proof, every experiment, every
figure, reproducible from the code in it — written for a reader who wants the
exact math.

**Live site:**
[charith-reddy-pareddy.github.io/soccer-markov-nash](https://charith-reddy-pareddy.github.io/soccer-markov-nash/)
— the write-up, plus an
[interactive board explorer](https://charith-reddy-pareddy.github.io/soccer-markov-nash/explorer.html)
that places both players anywhere and solves that exact position live, in the
browser, from the same solve this repo's figures and tables are built from.

## The problem

A pure-first Nash-Q approach to a configurable family of two-player soccer
Markov games. The environment, reward, and transition rules are design
choices of this project ([docs/design.md](docs/design.md)); Littman's 1994
soccer game and the CS&nbsp;540 A10 geometry are two instances of the family.

> Which geometric configurations of the soccer Markov game make mixed
> strategies necessary, can those configurations be characterized
> analytically, and can a solver that checks for a pure equilibrium first
> exploit that structure?

**Full answer: [docs/report.pdf](docs/report.pdf)**, also as
[docs/report.html](docs/report.html) (open in any browser) and
[docs/report.md](docs/report.md). Everything below is a compressed summary;
the report has the proofs, the full experiment grid, and the discussion.

## Two contributions

- **Algorithmic** &mdash; a pure-first hybrid Nash-Q backup: check each stage
  game for a pure saddle (`O(A²)`, no LP), take its value when it exists, fall
  back to the LP only where it does not. Exact everywhere; **it eliminates LP
  work on 96&ndash;100% of stage games** (a property of the game). The observed
  wall-clock speedup is reported separately.
- **Empirical structural characterization** &mdash; where the no-pure-saddle
  region is and what switches it on, over the tested parameter grid (not a
  theorem). Mixing appears under Littman's random move *order*, with a
  multi-cell goal mouth and a defender that cannot cover both scoring lanes.

## Four headline results

1. **The switch is the collision rule, not randomness.** A deterministic or
   coin-flip contest resolution leaves a pure saddle at every one of the
   238&thinsp;000 (state × step) stage games. Littman's random *move order* --
   the same joint action resolving differently depending on who moves first
   -- produces **94 of 2380 stage games (γ = 0.9) with no pure saddle**, on a
   7×5, 3-cell-goal board. Within the tested Littman-family boards, every
   one-cell goal produced zero such states, while every tested wider goal
   produced at least one; a different transition family (movement `slip`,
   this project's own `tackle` rule) breaks that switch even on a one-cell
   goal ([generalize.md](docs/generalize.md)) -- an empirical result scoped
   to the tested family, not a theorem about goal width in general.
2. **The pure-first hybrid is exact and (almost) free.** It reproduces the
   all-LP value function to `4e-16` while calling the LP on only 3.95% of
   states (~25× fewer per sweep); mirror symmetry halves the remaining work.
3. **94 no-pure-saddle states are not 94 different situations.** They
   canonicalize under the board mirror to 47 pairs and collapse again, by
   carrier-frame geometry, to just **8 canonical templates** -- and every one
   of them occurs with the two players at Manhattan distance 1 or 2 (40 at
   distance 1, 54 at distance 2, **zero** at distance ≥ 3). Distance to the
   *opponent*, not distance to the goal, is what forces a guess. Of the 94,
   **64 have a uniquely forced mix**; the rest (30) are degenerate LP
   outputs -- a reported split that is one point on a larger equally-valid
   equilibrium face, not a number the game forces
   ([docs/templates.md](docs/templates.md), [docs/degeneracy.md](docs/degeneracy.md)).
4. **Randomization matters because it prevents exploitation, not because it
   wins every matchup.** A minimax policy exploits weak opponents *and*
   survives a challenger built specifically to beat it; every deterministic
   policy tested -- including a hand-scripted one that beats a random
   opponent ~76% of the time -- does one or the other, never both
   ([docs/tournament.md](docs/tournament.md)). Patching a greedy policy with
   the exact Nash mix at *only* the 94 no-pure-saddle states recovers 43% of
   its robustness gap to minimax against a freshly built challenger
   ([docs/tournament_deepdive.md](docs/tournament_deepdive.md)) -- direct,
   causal evidence that it's predictability at those specific states, not
   overall play quality, that a challenger exploits.

**Neural Nash-Q is validation, not the headline.** With the full stochastic
transition distribution and exact minimax backups, a network can learn this
game -- but a policy net that names the right move 96&ndash;99% of the time
can still be the most exploitable of the baselines tested; a network with
worse action accuracy that hedges correctly where it counts is not
(`nash_dqn_random.py --seeds 5`,
[docs/report.md](docs/report.md) §8: *"action accuracy is not equilibrium
accuracy"*).

**Proof status.** Claims A and B below (deterministic game) are exact,
machine-checked results. The goal-width switch (headline result 1) is an
*empirical* result over the tested parameter grid. The single-cell case has a
closed-form, machine-verified defender strategy, but a board-size-free proof
of the carrier's half is still open -- see the claim table in
[docs/assumptions.md](docs/assumptions.md) and the "empirical / proved /
open" distinction in [docs/report.md](docs/report.md) §10.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
make lint            # ruff check
make test             # fast suite; `make test-all` adds the slow solver runs
make reproduce-core   # regenerate only the headline figures/tables above (~5 min)
make report           # docs/report.pdf from docs/report.html
```

`make experiments` regenerates the full `experiments/*.csv` sweep (slower,
includes the board sweep); `make reproduce-core` is the fast path for just
what this README and the report's core sections cite. CI
(`.github/workflows/tests.yml`) runs `ruff check` and the full `pytest` on
every push and pull request; `make coverage` reports line coverage (99%).

On very recent macOS builds the PyPI SciPy wheel can fail to load
(`_spropack.so` dyld error). If so, create the venv against a working
interpreter instead: `python -m venv --system-site-packages .venv`.

## Repository map

| path | what |
|---|---|
| `soccer_nash/game.py` | the Markov game (`SoccerGame`, `A10SoccerGame`) |
| `soccer_nash/matrix_games.py` | zero-sum stage solvers: pure saddle + LP minimax |
| `soccer_nash/nash_q.py` | Nash Q-iteration (`pure` / `mixed` / `hybrid`, VI and PI) |
| `soccer_nash/support_enum.py`, `markov_game.py` | all equilibria / general-sum extension |
| `soccer_nash/numerics.py` | value bracket, stage-game classification, rounding |
| `soccer_nash/certificate.py` | per-stage-game pure/mixed certificate, no LP |
| `soccer_nash/geometry.py`, `tree.py` | state features and the mixed-state decision tree |
| `soccer_nash/symmetry.py` | mirror states, symmetric solve |
| `soccer_nash/render.py`, `viz.py` | draw a state / policy fans / value heatmaps as SVG |
| `soccer_nash/a10.py`, `mlp.py`, `opponents.py`, `best_response.py` | A10 deliverables |
| `soccer_nash/shaping.py`, `exploit.py`, `nash_dqn.py` | reward shaping, self-play, neural Nash-Q |
| `soccer_nash/policy_gradient.py` | self-play REINFORCE, checked against the exact solver |
| `scripts/` | one entry point per analysis; `make <name>` runs most of them |
| `docs/*.md` | one page per topic, cross-linked from the report and `docs/README.md` |
| `experiments/*.csv` | committed sweep outputs the report tables read from |

Full script-by-script and doc-by-doc index: [docs/methods.md](docs/methods.md)
(the experiments table) and [docs/README.md](docs/README.md).

## The A10 collision rules: specified vs. interpreted

This repo's `A10SoccerGame` follows the assignment
([CS&nbsp;540 A10](https://pages.cs.wisc.edu/~yw/CS540S26A10.html)) where it
speaks, and makes an explicit, documented choice everywhere it does not. The
A10 page directly specifies: a 7×5 grid; two players choosing `U D L R`
simultaneously; "if the two players try to occupy the same square, only the
player with the ball will move to that square, and the other player will get
the ball"; "if the two players try to swap squares, they will swap, and the
other player will get the ball"; a tie after 100 steps; `+1` / `-1` / `0`
reward, undiscounted. It does **not** specify what happens when a carrier
moves into a *stationary* opponent, when a non-carrier bumps into the
carrier, which rows count as the goal mouth, or (in the public version of the
page) this project's own netID's start positions -- those are this repo's
interpretations, made explicit rather than presented as uniquely mandated.
**[docs/assumptions.md](docs/assumptions.md)** is the full quoted-vs-interpreted
table, which sub-cases are load-bearing for the headline result (the goal-row
choice is), and what would need re-measuring if course staff intended
something different.

The `random` and `coinflip` move-resolution rules, and every reward/transition
variant beyond that (`tackle`, `slip`, `blend`, `territory`), are this
project's own research extensions -- not part of the A10 assignment at all;
see [docs/design.md](docs/design.md).

## License

MIT &mdash; see [LICENSE](LICENSE).
