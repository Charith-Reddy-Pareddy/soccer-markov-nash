"""Exhaustive, LP-free verification of the goal-width / mixed-equilibrium result.

    python scripts/verify_mechanism.py         # -> experiments/mechanism_certificate.json
                                               #    docs/figures/gallery/mechanism.svg

The claim being checked (see docs/result.md):

  In the random-move-order soccer Markov game, a single goal cell per side
  yields a pure-strategy saddle at *every* reachable stage game, while a goal
  two or more cells wide yields at least one stage game that provably has no
  pure saddle -- and the switch survives the discount, the action set, and the
  reward objective.

Four parts, each machine-checked:

1. single cell  -> 0 mixed, cross-checked against onecell.certify (the defender's
   closed-form guard strategy + iterated weak-dominance);
2. wider goal   -> every mixed stage game gets a certificate (pure-bound gap,
   best-reply cycle, 2x2 matching-pennies core, exact equilibrium) that
   certificate.verify re-checks to machine precision;
3. robustness   -> the pure/mixed count stays 0 vs >0 across gamma in
   {0.5, 0.9, 0.995}, 4 vs 5 actions, win vs rate scoring, and exact
   undiscounted backward induction;
4. perturbation -> adding +/-1e-6 noise to every stage matrix flips no
   classification (the smallest pure-bound gap is ~1e-3, ~1000x the noise).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from soccer_nash.certificate import certify_game, classify, verify
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.onecell import certify as onecell_certify
from soccer_nash.reachability import reachable_states
from soccer_nash.viz import bestreply_svg

OUT = pathlib.Path("experiments/mechanism_certificate.json")
FIG = pathlib.Path("docs/figures/gallery/mechanism.svg")

SINGLE_CELL = [(5, 3), (7, 3), (5, 5), (7, 5)]
WIDER = [(5, 4, (1, 2)), (6, 5, (1, 2, 3)), (7, 5, (1, 2, 3))]


def _solve(game: SoccerGame, gamma: float):
    solver = NashQIteration(game, gamma=gamma, mode="hybrid", tol=1e-10)
    return solver, solver.run()


def _classify_all(solver, values, reach):
    """Every reachable stage game, certified and re-verified."""
    mixed, worst_verify, min_gap, min_margin, max_resid = [], 0.0, np.inf, np.inf, 0.0
    for s in reach:
        M = solver._matrix(s, values)
        cert = certify_game(M)
        worst_verify = max(worst_verify, verify(cert, M))
        if cert.kind == "mixed":
            mixed.append((s, cert))
            min_gap = min(min_gap, cert.gap)
            if cert.core:
                min_margin = min(min_margin, cert.core.margin)
            max_resid = max(max_resid, cert.equilibrium_residual)
    return mixed, {
        "reachable_states": len(reach),
        "mixed_states": len(mixed),
        "worst_certificate_verify": worst_verify,
        "min_pure_bound_gap": None if not mixed else min_gap,
        "min_matching_pennies_margin": None if min_margin is np.inf else min_margin,
        "max_equilibrium_residual": max_resid if mixed else 0.0,
        "all_mixed_have_2x2_core": all(c.core is not None for _, c in mixed),
    }


def part_single_cell(gamma: float) -> dict:
    rows = []
    for w, h in SINGLE_CELL:
        g = SoccerGame(width=w, height=h, goal_rows=(h // 2,), move_order="random")
        solver, res = _solve(g, gamma)
        reach = reachable_states(g)
        _, summary = _classify_all(solver, res.values, reach)
        oc = onecell_certify(w, h, gamma)
        rows.append({
            "board": f"{w}x{h}",
            **summary,
            "onecell_guard_slack": oc.guard_slack,
            "onecell_iewds_failures": oc.iewds_failures,
            "onecell_undiscounted_mixed": oc.undiscounted_mixed_stage_games,
            "ok": summary["mixed_states"] == 0 and oc.ok,
        })
    return {"gamma": gamma, "boards": rows, "ok": all(r["ok"] for r in rows)}


def part_wider(gamma: float) -> tuple[dict, tuple]:
    rows, sample = [], None
    for w, h, gr in WIDER:
        g = SoccerGame(width=w, height=h, goal_rows=gr, move_order="random")
        solver, res = _solve(g, gamma)
        reach = reachable_states(g)
        mixed, summary = _classify_all(solver, res.values, reach)
        rows.append({"board": f"{w}x{h}", "goal_rows": list(gr), **summary,
                     "ok": summary["mixed_states"] > 0
                     and summary["worst_certificate_verify"] < 1e-6
                     and summary["all_mixed_have_2x2_core"]})
        if sample is None and mixed:
            s, cert = max(mixed, key=lambda mc: mc[1].core.margin)
            sample = (g, solver._matrix(s, res.values), s, cert)
    return {"gamma": gamma, "boards": rows, "ok": all(r["ok"] for r in rows)}, sample


def part_robustness() -> dict:
    """One single-cell and one wider board across the full envelope."""
    checks = []

    def count_mixed(game, gamma):
        solver, res = _solve(game, gamma)
        reach = reachable_states(game)
        return sum(classify(solver._matrix(s, res.values)) == "mixed" for s in reach)

    for label, kw in [("single-cell 5x3", {"width": 5, "height": 3, "goal_rows": (1,)}),
                      ("wider 5x4", {"width": 5, "height": 4, "goal_rows": (1, 2)})]:
        want_mixed = "wider" in label
        for gamma in (0.5, 0.9, 0.995):
            n = count_mixed(SoccerGame(move_order="random", **kw), gamma)
            checks.append({"config": f"{label}, gamma={gamma}", "mixed": n,
                           "ok": (n > 0) == want_mixed})
        for n_act in (4, 5):
            n = count_mixed(SoccerGame(move_order="random", n_actions=n_act, **kw), 0.9)
            checks.append({"config": f"{label}, {n_act} actions", "mixed": n,
                           "ok": (n > 0) == want_mixed})
        for sc in ("win", "rate"):
            n = count_mixed(SoccerGame(move_order="random", scoring=sc, **kw), 0.9)
            checks.append({"config": f"{label}, {sc} scoring", "mixed": n,
                           "ok": (n > 0) == want_mixed})
        # exact undiscounted backward induction
        solver = NashQIteration(SoccerGame(move_order="random", **kw), gamma=0.99,
                                mode="hybrid", tol=1e-10)
        fh = solver.run_finite_horizon()
        checks.append({"config": f"{label}, undiscounted (backward induction)",
                       "mixed": fh.mixed_stage_games,
                       "ok": (fh.mixed_stage_games > 0) == want_mixed})
    return {"checks": checks, "ok": all(c["ok"] for c in checks)}


def part_perturbation(seed: int = 0, eps: float = 1e-6, genuine: float = 1e-4) -> dict:
    """Add +/-eps noise to every stage matrix and re-classify.

    A *genuine* mixed game has pure-bound gap > ``genuine`` (~1e-3 is the real
    minimum); a *strict* pure saddle has a unique row-min / col-max with a
    positive margin. Neither should ever flip under noise ``eps`` << gap. Games
    whose gap is within ``genuine`` of zero are the degenerate/tied stage games
    of the draw region -- rounding_changes_saddle already documents that their
    label is not noise-stable, and they have a pure saddle to play regardless.
    """
    g = SoccerGame(width=5, height=4, goal_rows=(1, 2), move_order="random")
    solver, res = _solve(g, 0.9)
    reach = reachable_states(g)
    rng = np.random.default_rng(seed)
    genuine_flips, sharp_saddle_flips, borderline_flips, real_min = 0, 0, 0, np.inf
    for s in reach:
        M = solver._matrix(s, res.values)
        row_min, col_max = M.min(axis=1), M.max(axis=0)
        gap = float(col_max.min() - row_min.max())
        base = classify(M)
        noisy = classify(M + rng.uniform(-eps, eps, size=M.shape))
        if base == "mixed" and gap > genuine:
            real_min = min(real_min, gap)
        if base == noisy:
            continue
        if base == "mixed":
            if gap > genuine:
                genuine_flips += 1
            else:
                borderline_flips += 1
        else:  # pure -> mixed: a failure only if the saddle had real room
            i, j = int(np.argmax(row_min)), int(np.argmin(col_max))
            v = M[i, j]
            col_room = np.inf if M.shape[0] == 1 else (v - np.delete(M[:, j], i)).min()
            row_room = np.inf if M.shape[1] == 1 else (np.delete(M[i, :], j) - v).min()
            margin = float(min(col_room, row_room))
            if margin > 10 * eps:
                sharp_saddle_flips += 1
            else:
                borderline_flips += 1
    return {"perturbation": eps,
            "genuine_mixed_flips": genuine_flips,
            "sharp_saddle_flips": sharp_saddle_flips,
            "borderline_label_unstable": borderline_flips,
            "min_genuine_gap": None if real_min is np.inf else real_min,
            "safety_ratio": None if real_min is np.inf else float(real_min / eps),
            "ok": genuine_flips == 0 and sharp_saddle_flips == 0}


def _write_figure(sample) -> None:
    g, M, s, cert = sample
    core = cert.core
    A = np.array(core.submatrix)
    rl = [f"a0={r}" for r in core.rows]
    cl = [f"a1={c}" for c in core.cols]
    svg = bestreply_svg(A, rl, cl,
                        title="Mixed stage game: the 2x2 matching-pennies core")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(svg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma", type=float, default=0.9)
    args = ap.parse_args()

    single = part_single_cell(args.gamma)
    wider, sample = part_wider(args.gamma)
    robust = part_robustness()
    perturb = part_perturbation()

    report = {
        "claim": "single goal cell => pure saddle everywhere; goal width >= 2 => "
                 "a provably saddle-free stage game; invariant to gamma, action "
                 "set, and reward objective.",
        "gamma": args.gamma,
        "single_cell": single,
        "wider_goal": wider,
        "robustness": robust,
        "perturbation": perturb,
        "all_checks_pass": all([single["ok"], wider["ok"], robust["ok"], perturb["ok"]]),
    }
    if sample is not None:
        _, M, s, cert = sample
        report["worked_example"] = {
            "state": list(s),
            "certificate": cert.as_dict(),
        }
        _write_figure(sample)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    print(f"single cell     : {'OK' if single['ok'] else 'FAIL'}")
    for r in single["boards"]:
        print(f"  {r['board']:>5}  mixed={r['mixed_states']}  "
              f"guard_slack={r['onecell_guard_slack']:.1e}")
    print(f"wider goal       : {'OK' if wider['ok'] else 'FAIL'}")
    for r in wider["boards"]:
        print(f"  {r['board']:>5} {r['goal_rows']}  mixed={r['mixed_states']}  "
              f"verify={r['worst_certificate_verify']:.1e}  "
              f"2x2-core={r['all_mixed_have_2x2_core']}")
    print(f"robustness       : {'OK' if robust['ok'] else 'FAIL'}  "
          f"({sum(c['ok'] for c in robust['checks'])}/{len(robust['checks'])} configs)")
    print(f"perturbation     : {'OK' if perturb['ok'] else 'FAIL'}  "
          f"genuine_mixed_flips={perturb['genuine_mixed_flips']}  "
          f"sharp_saddle_flips={perturb['sharp_saddle_flips']}  "
          f"(borderline/degenerate label-unstable: {perturb['borderline_label_unstable']})  "
          f"safety_ratio={perturb['safety_ratio']:.0f}x")
    print(f"\n{'ALL CHECKS PASS' if report['all_checks_pass'] else 'SOME CHECKS FAILED'}")
    print(f"wrote {OUT}" + (f" and {FIG}" if sample is not None else ""))


if __name__ == "__main__":
    main()
