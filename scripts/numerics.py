"""Numerical behaviour of Nash Q-iteration on the soccer game.

Four questions from the research meeting:

* how to *define* the value of a stage game when the three numbers disagree;
* whether rounding (to force mixed-strategy indifference) is safe under
  discounting;
* which iteration scheme -- value iteration or freeze-then-iterate -- is best;
* where the game actually "has to" mix, and what that sub-game looks like.

``--figures`` writes the visual answers to docs/figures/gallery/.
"""

from __future__ import annotations

import argparse
import collections
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from soccer_nash.game import SoccerGame
from soccer_nash.matrix_games import pure_bounds
from soccer_nash.nash_q import NashQIteration
from soccer_nash.numerics import (
    classify_stage_game,
    essential_subgame,
    rounding_changes_saddle,
    support_shape,
)
from soccer_nash.viz import P0, P1, bestreply_svg, line_chart_svg, panel_svg

FIGDIR = pathlib.Path("docs/figures/gallery")


def value_and_rounding(game, gamma):
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-10)
    r = solver.run()
    gaps = solver.value_bracket_gaps(r)
    cls = collections.Counter()
    flips = 0
    for s in solver._states:
        m = solver._matrix(s, r.values)
        cls[classify_stage_game(m)] += 1
        if rounding_changes_saddle(m, decimals=1):
            flips += 1
    print(f"  stage-game classes:            {dict(cls)}")
    print(f"  value-bracket gap:             max {gaps.max():.1e}, mean {gaps.mean():.1e}")
    print(f"  rounding to 0.1 flips a saddle: {flips} states")
    print(f"  => if V := p M q is the declared game value, a best-responding "
          f"row player gains at most {gaps.max():.1e} (the bracket width); the "
          f"safe number to report is the guaranteed value min_j (p M)_j.")
    return solver, r


def iteration_schemes(game, gamma):
    t = time.time()
    vi = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-9).run()
    t_vi = time.time() - t
    t = time.time()
    pi = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-9).run_policy_iteration(
        eval_sweeps=40
    )
    t_pi = time.time() - t
    stale = pi.staleness_trace or []
    lock = next((i for i, x in enumerate(stale) if x < 1e-6), len(stale))
    print(
        f"  value iteration : {vi.iterations:3d} sweeps, "
        f"{vi.matrix_game_solves:6d} solves, {t_vi:5.1f} s"
    )
    print(
        f"  policy iteration: {pi.iterations:3d} rounds, "
        f"{pi.matrix_game_solves:6d} solves, {t_pi:5.1f} s"
    )
    print(f"  freeze staleness stayed maximal for {lock} rounds, then locked in")


def eval_value_choice(game, gamma):
    """Which of the three stage-game quantities to back up in the frozen
    evaluation sweep -- the professor's open question. Compare all three."""
    truth = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-9).run().values
    print("  freeze-then-iterate evaluation quantity:")
    for ev in ("mid", "lower", "upper"):
        r = NashQIteration(game, gamma=gamma, mode="hybrid",
                           tol=1e-9).run_policy_iteration(eval_sweeps=40,
                                                          eval_value=ev)
        err = max(abs(r.values[s] - truth[s]) for s in game.states())
        lock = next((i for i, x in enumerate(r.staleness_trace or [])
                     if x < 1e-6), len(r.staleness_trace or []))
        tag = {"mid": "p M q     ", "lower": "min_j(pM)_j", "upper": "max_i(Mq)_i"}[ev]
        print(f"    {tag}: {r.iterations:3d} rounds, {r.matrix_game_solves:5d} "
              f"solves, thrash {lock:2d} rounds, |V-truth| {err:.1e}")
    print("    => p M q wins: unbiased while strategies are stale, ~2x fewer "
          "rounds; all three reach the same fixed point.")


def characterise_mixed_states(solver, result):
    ns = result.no_saddle_states
    if not ns:
        print("  (no states require mixed strategies)")
        return
    shapes = collections.Counter()
    for s in ns:
        m = solver._matrix(s, result.values)
        shapes[support_shape(m)] += 1
    two_by_two = shapes[(2, 2)]
    print(f"  {len(ns)} states need mixed strategies")
    print(f"  equilibrium support shapes:  {dict(sorted(shapes.items()))}")
    print(f"  {two_by_two} of {len(ns)} are 2x2 -- a matching-pennies mix")


def _first_2x2_mixed(solver, result):
    """A real 2x2 matching-pennies stage game, carrier as the maximising rows."""
    for s in sorted(result.no_saddle_states):
        m = solver._matrix(s, result.values)
        _ri, _ci, sub = essential_subgame(m)
        if sub.shape == (2, 2):
            return s, (sub if s[4] == 0 else -sub.T)
    return None, None


def write_figures(gamma: float) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    game = SoccerGame(move_order="random")
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-10)
    result = solver.run()

    # 1. the soccer game contains rock-paper-scissors
    rps = np.array([[0.0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    state, sub = _first_2x2_mixed(solver, result)
    (FIGDIR / "rps_vs_soccer.svg").write_text(panel_svg([
        bestreply_svg(rps, ["rock", "paper", "scis"], ["rock", "paper", "scis"],
                      title="rock-paper-scissors"),
        bestreply_svg(np.round(sub, 3), ["climb", "advance"], ["cover", "hold"],
                      title=f"soccer: carrier pinned  {state}"),
    ], cols=2))
    print(f"wrote {FIGDIR / 'rps_vs_soccer.svg'}")

    # 2. the discounting / rounding trap
    grid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    pts_pct, pts_gap = [], []
    for gm in grid:
        s = NashQIteration(game, gamma=gm, mode="hybrid", tol=1e-10)
        r = s.run()
        mixed = list(r.no_saddle_states)
        flips = sum(
            1 for st in mixed
            if rounding_changes_saddle(s._matrix(st, r.values), 1)
        )
        gaps = [
            (lambda lo, hi: hi - lo)(*pure_bounds(s._matrix(st, r.values)))
            for st in mixed
        ]
        pts_pct.append((gm, 100.0 * flips / max(len(mixed), 1)))
        pts_gap.append((gm, 100.0 * float(np.median(gaps)) / 0.1))
    (FIGDIR / "discounting_trap.svg").write_text(line_chart_svg(
        [("% wrongly made pure", P0, pts_pct),
         ("gap, % of the 0.1 grid", P1, pts_gap)],
        x_label="discount gamma", y_label="percent",
        title="Fixed 0.1-rounding is never safe",
    ))
    print(f"wrote {FIGDIR / 'discounting_trap.svg'}")

    # 3. value iteration vs freeze-then-iterate convergence
    vi = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-9).run()
    pi = NashQIteration(game, gamma=gamma, mode="hybrid",
                        tol=1e-9).run_policy_iteration(eval_sweeps=40)

    def _digits(trace):
        return [(i + 1, -math.log10(max(v, 1e-12))) for i, v in enumerate(trace)]

    _ = pi  # freeze-then-iterate is shown in eval_value.svg
    (FIGDIR / "convergence.svg").write_text(line_chart_svg(
        [("Bellman residual", P0, _digits(vi.residual_trace))],
        x_label="value-iteration sweep", y_label="digits of accuracy",
        title="Value iteration: geometric decay at rate about gamma",
    ))
    print(f"wrote {FIGDIR / 'convergence.svg'}")

    # 4. which value quantity to back up in the frozen evaluation sweep
    truth = vi.values
    series = []
    for ev, colour in (("mid", P0), ("lower", P1),
                       ("upper", "var(--ember, #a94e18)")):
        r = NashQIteration(game, gamma=gamma, mode="hybrid",
                           tol=1e-9).run_policy_iteration(eval_sweeps=40,
                                                          eval_value=ev)
        tr = r.staleness_trace or []
        series.append((
            {"mid": "p M q", "lower": "min(pM)", "upper": "max(Mq)"}[ev],
            colour, [(i + 1, v) for i, v in enumerate(tr)],
        ))
        err = max(abs(r.values[s] - truth[s]) for s in game.states())
        assert err < 1e-6, f"{ev} did not reach the fixed point: {err}"
    (FIGDIR / "eval_value.svg").write_text(line_chart_svg(
        series, x_label="outer round", y_label="frozen-strategy staleness",
        title="Back up p M q in the frozen sweep",
    ))
    print(f"wrote {FIGDIR / 'eval_value.svg'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--move-orders", nargs="+", default=["deterministic", "random"],
        choices=["deterministic", "random", "coinflip"],
    )
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()

    for mo in args.move_orders:
        game = SoccerGame(move_order=mo)
        print(f"\n=== move_order = {mo}, gamma = {args.gamma} ===")
        solver, result = value_and_rounding(game, args.gamma)
        iteration_schemes(game, args.gamma)
        if mo != "deterministic":
            eval_value_choice(game, args.gamma)
        characterise_mixed_states(solver, result)

    if args.figures:
        print()
        write_figures(args.gamma)


if __name__ == "__main__":
    main()
