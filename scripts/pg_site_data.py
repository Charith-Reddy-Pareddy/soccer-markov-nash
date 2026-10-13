# ruff: noqa: E501
"""Results of the policy-gradient experiments as one JSON file for the site's
Policy gradient tab, so every number on that page comes from a run.

    python scripts/pg_site_data.py        # writes site/src/pgResults.json
"""

from __future__ import annotations

import csv
import glob
import json
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"

LEARNERS = [
    ("REINFORCE", "standard", "reinforce", "selfplay"),
    ("REINFORCE", "fictitious play", "reinforce", "fictitious"),
    ("A2C", "standard", "a2c", "selfplay"),
    ("A2C", "fictitious play", "a2c", "fictitious"),
    ("PPO", "standard", "ppo", "selfplay"),
    ("PPO", "fictitious play", "ppo", "fictitious"),
]
EXTRA = [
    ("A2C, exact critic", "standard", "a2c_exact", "selfplay"),
    ("A2C, exact critic", "fictitious play", "a2c_exact", "fictitious"),
]
NAMES = {"reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}


def read(path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def mean(rows, key) -> float:
    return st.mean(float(r[key]) for r in rows)


def wtl(rows, opp) -> list[float]:
    return [round(mean(rows, f"row_{k}_vs_{opp}"), 6) for k in ("win", "tie", "loss")]


def wtl_column(rows, opp) -> list[float]:
    """Win / tie / loss of the same learners playing the column seat, against the best response
    to the column policy."""
    return [round(mean(rows, f"col_{k}_vs_{opp}"), 6) for k in ("win", "tie", "loss")]


def summary(rows) -> dict:
    seeds = [float(r["exploitability"]) for r in rows]
    return {
        "seeds": len(rows),
        "exploitability": {
            "mean": round(st.mean(seeds), 6),
            "sd": round(st.stdev(seeds), 6) if len(seeds) > 1 else 0.0,
            "runs": seeds,
        },
        "vs_random": wtl(rows, "random"),
        "vs_nash": wtl(rows, "nash"),
        "vs_best_response": wtl(rows, "br"),
        "vs_best_response_column": wtl_column(rows, "br"),
        "mirror_gap": round(mean(rows, "mirror_gap_mean"), 6),
    }


def build() -> dict:
    main = read(EXP / "pg_finite_a10.csv")
    long_rows = [r for p in sorted(glob.glob(str(EXP / "pg_finite_a10_long_*.csv")))
                 for r in read(p) if r["algo"] != "exact"]
    learners, longer = [], []
    for label, training, algo, mode in LEARNERS:
        rs = [r for r in main if r["algo"] == algo and r["mode"] == mode]
        learners.append({"label": label, "training": training, **summary(rs)})
        lr = sorted((r for r in long_rows if r["algo"] == algo and r["mode"] == mode),
                    key=lambda r: int(r["seed"]))
        if lr:
            longer.append({
                "label": label, "training": training,
                "short_mean": round(mean(rs, "exploitability"), 6),
                "short_vs_nash": wtl(rs, "nash"), "short_vs_best_response": wtl(rs, "br"),
                "vs_best_response": wtl(lr, "br"),
                "runs": [round(float(r["exploitability"]), 6) for r in lr],
                "vs_random": wtl(lr, "random"), "vs_nash": wtl(lr, "nash"),
            })
    fp_br = []
    for path in sorted(glob.glob(str(EXP / "pg_fp_br_*.csv"))):
        rs = read(path)
        fp_br.append({
            "label": NAMES[rs[0]["algo"]], "best_response_iterations": int(rs[0]["br_iters"]),
            "checkpoints": [{"round": int(r["round"]),
                             "exploitability": round(float(r["exploitability"]), 6),
                             "win_vs_best_response": round(float(r["row_win_vs_br"]), 6)}
                            for r in rs],
        })
    return {"exact": summary([r for r in main if r["algo"] == "exact"]),
            "learners": learners, "longer": longer, "fp_br": fp_br,
            "random": random_board(), "mixed": mixed_states(),
            "continuing": continuing_game(), "rps": rock_paper_scissors(),
            "variants": {b: variants(b) for b in VARIANT_FILES},
            "argmax": argmax_average(), "rps_nn": rps_networks(),
            "entropy": larger_entropy(), "choices": training_choices(),
            "curves": training_curves(), "related": RELATED,
            "more_seeds": more_seeds()}


def random_board() -> dict:
    """The same six learners on the random move-order board (3 seeds each)."""
    rows = [r for p in sorted(glob.glob(str(EXP / "pg_finite_random_*.csv"))) for r in read(p)]
    learners = []
    for label, training, algo, mode in LEARNERS:
        rs = [r for r in rows if r["algo"] == algo and r["mode"] == mode]
        learners.append({"label": label, "training": training, **summary(rs)})
    kick = json.loads((EXP / "pg_random_kickoff_value.json").read_text())
    return {"exact": summary([r for r in rows if r["algo"] == "exact"]), "learners": learners,
            "kickoff_value": kick["value"]}


VARIANT_FILES = {
    "random": ("pg_finite_random_{}.csv", "pg_variant_random_shared.csv",
               "pg_variant_random_trim10.csv"),
    "deterministic": ("pg_finite_a10.csv", "pg_variant_a10_shared.csv",
                      "pg_variant_a10_trim10.csv"),
    "continuing": ("pg_finite_rate_a10_{}_*.csv", "pg_variant_rate_a10_shared.csv",
                   "pg_variant_rate_a10_trim10.csv"),
}


def variants(board: str) -> list[dict] | None:
    """A2C and PPO with separate networks (baseline), one shared network, and the last 10 steps
    of each episode left out of the loss: win rate against the best response and share of ties
    with the exact Nash policy, mean and per seed."""
    out = []
    for algo in ("a2c", "ppo"):
        for training, mode in (("standard", "selfplay"), ("fictitious play", "fictitious")):
            entry = {"label": NAMES[algo], "training": training}
            for key, pattern in zip(("baseline", "shared", "trimmed"), VARIANT_FILES[board]):
                paths = sorted(glob.glob(str(EXP / pattern.format(algo))))
                if not paths:
                    return None
                rs = [r for path in paths for r in read(path)
                      if r["algo"] == algo and r["mode"] == mode]
                br = [float(r["row_win_vs_br"]) for r in rs]
                tie = [float(r["row_tie_vs_nash"]) for r in rs]
                entry[key] = {"mean": _avg(br), "runs": br, "tie_nash_mean": _avg(tie),
                              "tie_nash_runs": tie, "vs_random": wtl(rs, "random")}
            out.append(entry)
    return out


def rps_networks() -> dict | None:
    """Rock-paper-scissors with neural-network players (experiments/rps_nn.json)."""
    path = EXP / "rps_nn.json"
    return json.loads(path.read_text()) if path.exists() else None


# board -> (name in output files, baseline file for standard / fictitious, argmax file)
BOARD_FILES = {
    "random": ("random", "pg_finite_random_{algo}.csv", "pg_variant_random_argmax_{algo}.csv"),
    "deterministic": ("a10", "pg_finite_a10.csv", "pg_variant_a10_argmax_{algo}.csv"),
    "continuing": ("rate_a10", "pg_finite_rate_a10_{algo}_{mode}.csv",
                   "pg_variant_rate_a10_argmax_{algo}.csv"),
}


def _runs(board: str, algo: str, mode: str, entropy: float = 0.01) -> list[dict]:
    """The result rows of one learner (3 seeds) on a board, at the usual or the larger entropy
    bonus; ``mode`` is selfplay, fictitious or fictitious_argmax."""
    name, base, argm = BOARD_FILES[board]
    if entropy == 0.01:
        pattern = argm if mode == "fictitious_argmax" else base
    else:
        pattern = f"pg_entropy_{name}_{mode}_{{algo}}.csv"
    paths = sorted(glob.glob(str(EXP / pattern.format(algo=algo, mode=mode))))
    return [r for path in paths for r in read(path) if r["algo"] == algo and r["mode"] == mode]


def _stats(rs: list[dict]) -> dict:
    br = [float(r["row_win_vs_br"]) for r in rs]
    tie = [float(r["row_tie_vs_nash"]) for r in rs]
    rand = [float(r["row_win_vs_random"]) for r in rs]
    return {"mean": _avg(br), "runs": br, "tie_nash_mean": _avg(tie), "tie_nash_runs": tie,
            "rand_mean": _avg(rand), "rand_runs": rand,
            "vs_random": wtl(rs, "random"), "vs_nash": wtl(rs, "nash"),
            "vs_best_response": wtl(rs, "br")}


def argmax_average() -> dict | None:
    """Fictitious play whose snapshots are pure (argmax) policies, against the usual
    fictitious play that averages softmax policies; REINFORCE, A2C and PPO on the three boards,
    three seeds each."""
    out = {}
    for board in BOARD_FILES:
        rows_out = []
        for algo in ("reinforce", "a2c", "ppo"):
            entry = {"label": NAMES[algo], "training": "fictitious play"}
            for key, mode in (("softmax", "fictitious"), ("argmax", "fictitious_argmax")):
                rs = _runs(board, algo, mode)
                if len(rs) < 3:
                    return out or None
                entry[key] = _stats(rs)
            rows_out.append(entry)
        out[board] = rows_out
    return out


MODE_NAMES = {"selfplay": "standard", "fictitious": "fictitious play",
              "fictitious_argmax": "fictitious play, argmax"}


def larger_entropy() -> dict | None:
    """The usual entropy bonus (0.01) against a larger one (0.2) for every learner, training
    scheme and board."""
    out = {}
    for board in BOARD_FILES:
        rows_out = []
        for algo in ("reinforce", "a2c", "ppo"):
            for mode in ("selfplay", "fictitious", "fictitious_argmax"):
                small, large = _runs(board, algo, mode), _runs(board, algo, mode, 0.2)
                if len(small) == 3 and len(large) == 3:
                    rows_out.append({"label": NAMES[algo], "training": MODE_NAMES[mode],
                                     "baseline": _stats(small), "larger": _stats(large)})
        if len(rows_out) == 9:
            out[board] = rows_out
    return out or None


STUDY_NAMES = {"random": "random", "deterministic": "a10", "continuing": "rate_a10"}
LEARNER_LABELS = {"a2c": "A2C", "ppo": "PPO", "reinforce": "REINFORCE", "a2c_exact": "A2C, exact critic"}


def _file_rows(pattern: str, algo: str, mode: str) -> list[dict]:
    paths = sorted(glob.glob(str(EXP / pattern)))
    return [r for path in paths for r in read(path) if r["algo"] == algo and r["mode"] == mode]


def _study(title: str, note: str, metric: str, reference, keys: list[tuple[str, str]],
           rows: list[tuple[str, str, dict[str, list[dict]]]]) -> dict | None:
    """One comparison: for every (label, training) row, the results of each setting in ``keys``
    (key, label); ``None`` until every setting has all 3 seeds."""
    out_rows = []
    for label, training, by_key in rows:
        entry = {"label": label, "training": training}
        for key, _ in keys:
            rs = by_key[key]
            if len(rs) < 3:
                return None
            entry[key] = _stats(rs)
        out_rows.append(entry)
    return {"title": title, "note": note, "metric": metric, "reference": reference,
            "keys": [list(k) for k in keys], "rows": out_rows}


def _trim_rows(board: str, algo: str, mode: str, trim: int) -> list[dict]:
    name = STUDY_NAMES[board]
    if trim == 0:
        return _runs(board, algo, mode)
    if trim == 10:
        return _file_rows(f"pg_variant_{name}_trim10.csv", algo, mode)
    return _file_rows(f"pg_study_{name}_trim{trim}_{mode}_{algo}.csv", algo, mode)


def training_choices() -> list[dict] | None:
    """Training choices compared on the usual setting: how many steps to leave out of the loss,
    how long to train per fictitious-play snapshot and in total, mini-batches and replay, the
    exact critic on every board, and averaging weights instead of policies."""
    rnd_ref = None
    studies = []
    modes = (("selfplay", "standard"), ("fictitious", "fictitious play"))
    algos = (("a2c", "A2C"), ("ppo", "PPO"))
    trims = [0, 5, 10, 20, 40]
    tkeys = [(f"t{t}", "all steps" if t == 0 else f"last {t} left out") for t in trims]
    studies.append(_study(
        "Leaving out the last steps: how many?",
        "Random move-order board; the last N of the 100 steps of every episode are left out of the loss.",
        "br", rnd_ref, tkeys,
        [(lab, tr, {f"t{t}": _trim_rows("random", a, m, t) for t in trims})
         for a, lab in algos for m, tr in modes]))
    studies.append(_study(
        "Leaving out the last steps in the continuing game",
        "Continuing game, standard training; share of games tied with the exact Nash policy.",
        "tie", None, tkeys,
        [(lab, "standard", {f"t{t}": _trim_rows("continuing", a, "selfplay", t) for t in trims})
         for a, lab in algos]))
    skeys = [("s1", "snapshot every iteration"), ("s5", "every 5 (used elsewhere)"),
             ("s25", "every 25"), ("s100", "every 100")]
    studies.append(_study(
        "How long to train between snapshots",
        "Random board, fictitious play, 2,000 iterations in total: the training budget of each best response.",
        "br", rnd_ref, skeys,
        [(lab, "fictitious play", {
            "s1": _file_rows(f"pg_study_random_snap1_fictitious_{a}.csv", a, "fictitious"),
            "s5": _runs("random", a, "fictitious"),
            "s25": _file_rows(f"pg_study_random_snap25_fictitious_{a}.csv", a, "fictitious"),
            "s100": _file_rows(f"pg_study_random_snap100_fictitious_{a}.csv", a, "fictitious")})
         for a, lab in algos]))
    studies.append(_study(
        "Five times as many iterations",
        "Random board: 10,000 iterations instead of 2,000.",
        "br", rnd_ref, [("i2000", "2,000 iterations"), ("i10000", "10,000 iterations")],
        [(lab, tr, {"i2000": _runs("random", a, m),
                    "i10000": _file_rows(f"pg_study_random_iter10000_{m}_{a}.csv", a, m)})
         for a, lab in algos for m, tr in modes]))
    studies.append(_study(
        "Mini-batches and replayed data",
        "Random board. Mini-batch: many gradient steps per iteration, each on 256 random samples. "
        "Replay: each update also uses the data of the 3 previous iterations.",
        "br", rnd_ref, [("base", "all data, one step"), ("mb", "mini-batches of 256"),
                        ("rp", "replay of 4 iterations")],
        [(lab, tr, {"base": _runs("random", a, m),
                    "mb": _file_rows(f"pg_study_random_mb256_{m}_{a}.csv", a, m),
                    "rp": _file_rows(f"pg_study_random_replay4_{m}_{a}.csv", a, m)})
         for a, lab in algos for m, tr in modes]))
    for board, metric, title in (("random", "br", "Random board"),
                                 ("deterministic", "tie", "Deterministic board")):
        name = STUDY_NAMES[board]
        studies.append(_study(
            f"The exact solver's values as the critic: {title.lower()}",
            "A2C with a learned critic against A2C with the exact solver's values as a frozen critic.",
            metric, None, [("learned", "learned critic"), ("exact", "exact critic")],
            [("A2C", tr, {"learned": _runs(board, "a2c", m),
                          "exact": _file_rows(f"pg_study_{name}_a2cexact_{m}.csv", "a2c_exact", m)})
             for m, tr in modes]))
    wkeys = [("pol", "average of policies"), ("w", "average of weights"),
             ("wl", "average of the output layer's weights")]
    wrows = []
    for a, lab in (("reinforce", "REINFORCE"), *algos):
        rs = _file_rows(f"pg_study_random_weights_{a}.csv", a, "fictitious")
        wrows.append((lab, "fictitious play", {
            "pol": rs, "w": [_prefixed(r, "wavg_") for r in rs],
            "wl": [_prefixed(r, "wlast_") for r in rs]}))
    studies.append(_study(
        "Averaging weights instead of policies",
        "Random board, fictitious play. Fictitious play averages policies; the other two build one "
        "network from the snapshots' weights (all layers, or only the output layer).",
        "br", rnd_ref, wkeys, wrows))
    studies = [s for s in studies if s]
    return studies or None


def _prefixed(row: dict, prefix: str) -> dict:
    """A result row whose ``row_*`` scores are those stored under ``prefix``."""
    out = dict(row)
    for k in list(row):
        if k.startswith(prefix):
            out[k[len(prefix):]] = row[k]
    return out


# two-sided 95% critical values of Student's t, by degrees of freedom
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
       2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
       2.052, 2.048, 2.045, 2.042]


def t95(df: float) -> float:
    return T95[max(0, min(len(T95), int(df)) - 1)]


def interval(values: list[float]) -> list[float]:
    """Mean and the ends of its 95% confidence interval (Student's t), kept within 0 and 1 because
    the values are shares of games."""
    n, m = len(values), st.mean(values)
    half = t95(n - 1) * st.stdev(values) / n ** 0.5
    return [round(m, 6), round(max(m - half, 0.0), 6), round(min(m + half, 1.0), 6)]


def difference(a: list[float], b: list[float]) -> list[float]:
    """Mean of ``b`` minus mean of ``a`` and the ends of its 95% interval (Welch)."""
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = t95(df) * (va + vb) ** 0.5
    d = st.mean(b) - st.mean(a)
    return [round(d, 6), round(d - half, 6), round(d + half, 6)]


MORE_SEEDS = 10
# condition -> (first-three-seeds file, mode, seed files of the extra runs)
MORE_CONDITIONS = {
    "a2c_fp_base": ("pg_finite_random_a2c.csv", "a2c", "fictitious", "pg_more_random_a2c_fp_base_s*.csv"),
    "a2c_fp_10k": ("pg_study_random_iter10000_fictitious_a2c.csv", "a2c", "fictitious", "pg_more_random_a2c_fp_10k_s*.csv"),
    "ppo_sp_base": ("pg_finite_random_ppo.csv", "ppo", "selfplay", "pg_more_random_ppo_sp_base_s*.csv"),
    "ppo_sp_replay4": ("pg_study_random_replay4_selfplay_ppo.csv", "ppo", "selfplay", "pg_more_random_ppo_replay4_s*.csv"),
}


def _more_rows(key: str) -> list[dict]:
    first, algo, mode, extra = MORE_CONDITIONS[key]
    rows = _file_rows(first, algo, mode) + _file_rows(extra, algo, mode)
    by_seed = {int(r["seed"]): r for r in rows}
    return [by_seed[k] for k in sorted(by_seed)]


def more_seeds() -> dict | None:
    """The two settings that looked best with 3 seeds (A2C fictitious play at 10,000 iterations and
    PPO standard training with replay of 4 iterations) and their baselines, on seeds 0-9: win rate
    against the best response from each kickoff seat and balanced, with 95% intervals."""
    conds = {}
    for key in MORE_CONDITIONS:
        rows = _more_rows(key)
        if len(rows) < MORE_SEEDS:
            return None
        ball = [float(r["row_win_vs_br"]) for r in rows]
        other = [float(r["col_win_vs_br"]) for r in rows]
        balanced = [(x + y) / 2 for x, y in zip(ball, other)]
        conds[key] = {"seeds": [int(r["seed"]) for r in rows], "ball": ball, "other": other,
                      "balanced": balanced, "ball_ci": interval(ball), "other_ci": interval(other),
                      "balanced_ci": interval(balanced),
                      "random": [float(r["row_win_vs_random"]) for r in rows],
                      "tie_nash": [float(r["row_tie_vs_nash"]) for r in rows]}
    pairs = []
    for label, base, var, variant in (("A2C, fictitious play", "a2c_fp_base", "a2c_fp_10k", "10,000 iterations"),
                                      ("PPO, standard", "ppo_sp_base", "ppo_sp_replay4", "replay of 4 iterations")):
        pairs.append({"label": label, "variant": variant, "baseline_key": base, "variant_key": var,
                      **{f"{m}_diff": difference(conds[base][m], conds[var][m]) for m in ("ball", "other", "balanced")}})
    return {"seeds": MORE_SEEDS, "conditions": conds, "pairs": pairs}


def training_curves() -> dict | None:
    """Win rate against the best response and the kickoff policy during training, standard
    against fictitious play (experiments/pg_curves_*.csv)."""
    kickoff_path = EXP / "pg_exact_kickoff.json"
    if not kickoff_path.exists():
        return None
    kick = json.loads(kickoff_path.read_text())
    out = {}
    for board, name in (("random", "random"), ("deterministic", "a10")):
        series = []
        for algo in ("a2c", "ppo"):
            for mode in ("selfplay", "fictitious"):
                rows = _file_rows(f"pg_curves_{name}_{mode}_{algo}.csv", algo, mode)
                seeds = sorted({int(r["seed"]) for r in rows})
                if len(seeds) < 3:
                    return out or None
                iters = sorted({int(r["iteration"]) for r in rows})
                best = int(max(range(4), key=lambda k: kick[board]["row"][k]))

                def at(seed, it, col, rows=rows):
                    return next(float(r[col]) for r in rows
                                if int(r["seed"]) == seed and int(r["iteration"]) == it)

                series.append({
                    "label": LEARNER_LABELS[algo], "training": MODE_NAMES[mode], "mode": mode,
                    "move": best,
                    "win_br": [_avg([at(sd, it, "win_br") for sd in seeds]) for it in iters],
                    "win_nash": [_avg([at(sd, it, "win_nash") for sd in seeds]) for it in iters],
                    "tie_nash": [_avg([at(sd, it, "tie_nash") for sd in seeds]) for it in iters],
                    "seed_win_br": [[at(sd, it, "win_br") for it in iters] for sd in seeds],
                    "current_move": [at(0, it, f"cur_{best}") for it in iters],
                    "aggregate_move": [at(0, it, f"agg_{best}") for it in iters]})
        out[board] = {"iterations": iters, "series": series,
                      "exact_move_probability": kick[board]["row"][best], "move": "UDLR"[best]}
    return out or None


RELATED = [
    {"title": "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games",
     "authors": "Heinrich and Silver", "year": 2016, "url": "https://arxiv.org/abs/1603.01121",
     "says": "Neural fictitious self-play (NFSP) learns approximate Nash equilibria of poker games "
             "end to end by combining fictitious self-play with deep reinforcement learning; "
             "ordinary reinforcement learning diverged on the small game (Leduc poker) where NFSP "
             "moved toward equilibrium."},
    {"title": "Policy Optimization in Zero-Sum Markov Games: Fictitious Self-Play Provably Attains "
              "Nash Equilibria",
     "authors": "", "year": None,
     "url": "https://astro.paperswithcode.com/paper/policy-optimization-in-zero-sum-markov-games",
     "says": "A smooth fictitious self-play whose best-response step is a PPO step with entropy "
             "regularization reaches a neighbourhood of a Nash equilibrium at a sublinear rate in "
             "zero-sum Markov games (a guarantee for the regularized method, not the exact "
             "equilibrium)."},
    {"title": "Policy Optimization for Markov Games: Unified Framework and Faster Convergence",
     "authors": "Zhang et al.", "year": 2022,
     "url": "https://proceedings.neurips.cc/paper_files/paper/2022/hash/"
            "8951f484e8242b7f74817fdc390dd954-Abstract.html",
     "says": "Each iteration updates the policy at every state with a matrix-game algorithm and then "
             "the values; the state-wise average policy converges to an approximate Nash "
             "equilibrium of two-player zero-sum Markov games."},
    {"title": "Fast Policy Extragradient Methods for Competitive Games with Entropy Regularization",
     "authors": "Cen, Wei and Chi", "year": 2024, "url": "https://jmlr.org/papers/v25/21-1205.html",
     "says": "Entropy regularization gives linear-rate convergence to the regularized equilibrium "
             "(including in zero-sum Markov games), and tuning the weight approaches an approximate "
             "Nash equilibrium of the unregularized game."},
    {"title": "Fictitious Play in Markov Games with Single Controller",
     "authors": "Sayin, Zhang and Ozdaglar", "year": 2022, "url": "https://arxiv.org/abs/2205.11389",
     "says": "A variant of fictitious play converges in identical-interest Markov games where one "
             "player controls the transitions; together with earlier results it covers both zero-sum "
             "and identical-interest Markov games."},
    {"title": "Fictitious play in zero-sum stochastic games",
     "authors": "Sayin, Parise and Ozdaglar", "year": 2022,
     "url": "https://deepai.org/publication/fictitious-play-in-zero-sum-stochastic-games",
     "says": "A fictitious-play variant that combines classical fictitious play with Q-learning for "
             "two-player zero-sum stochastic games."},
]


def _avg(values) -> float:
    return round(st.mean(values), 6)


def _kind(row) -> str:
    return "three-way mix" if sum(v > 1e-6 for v in row) >= 3 else "two-way mix"


def continuing_game() -> dict | None:
    """The deterministic board as a fixed-length game that restarts after every goal; a
    player wins a game by scoring more goals than the opponent in 100 steps."""
    paths = [q for q in sorted(glob.glob(str(EXP / "pg_finite_rate_a10_*.csv")))
             if "_shared" not in q and "_trim" not in q]
    if not paths:
        return None
    rows = [r for p in paths for r in read(p)]
    learners = []
    for label, training, algo, mode in [*LEARNERS, *EXTRA]:
        rs = [r for r in rows if r["algo"] == algo and r["mode"] == mode]
        if rs:
            learners.append({"label": label, "training": training, **summary(rs)})
    if len(learners) < len(LEARNERS) + len(EXTRA) or any(x["seeds"] < 3 for x in learners):
        return None  # published only once every learner has all 3 seeds
    return {"exact": summary([r for r in rows if r["algo"] == "exact"]), "learners": learners}


def rock_paper_scissors(rounds: int = 60) -> dict:
    """Best-response dynamics against fictitious play on rock-paper-scissors: the share of
    rock in the current play (cycles) against in the average of all play so far (settles)."""
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from soccer_nash.fictitious_play import best_response_dynamics, fp_step

    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    path = best_response_dynamics(rps, rounds - 1)
    cr, cc = np.zeros((1, 3)), np.zeros((1, 3))
    cr[0, 0] = cc[0, 0] = 1.0
    average = [1.0]
    for _ in range(rounds - 1):
        fp_step(rps[None], cr, cc)
        average.append(float(cr[0, 0] / cr[0].sum()))
    return {"rounds": rounds,
            "best_response": [1.0 if a == 0 else 0.0 for a, _ in path[:rounds]],
            "average": [round(v, 6) for v in average]}


def mixed_states() -> dict | None:
    """The learners at the random board's mixed states: wins from games started there, how
    close their probabilities are to the exact mix, and a few states shown in full."""
    path = EXP / "pg_mixed_states.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    learners, outputs = [], []
    for label, training, algo, mode in LEARNERS:
        rs = [r for r in d["runs"] if r["algo"] == algo and r["mode"] == mode]

        def triple(key, rs=rs):
            return [_avg(r["wins_from_mixed_starts"][key][i] for r in rs) for i in range(3)]

        learners.append({
            "label": label, "training": training, "seeds": len(rs),
            "vs_random": triple("random"), "vs_nash": triple("nash"),
            "vs_best_response": triple("br"),
            "tv_row": _avg(r["tv_row"] for r in rs), "tv_col": _avg(r["tv_col"] for r in rs),
            "regret_mean": _avg(r["regret_mean"] for r in rs),
            "regret_max": round(max(r["regret_max"] for r in rs), 6),
            "share_mixing_row": _avg(r["share_mixing_row"] for r in rs),
        })
        first = next(r for r in rs if r["seed"] == 0)
        outputs.append({"label": f"{label}, {training}",
                        "row": [e["row"] for e in first["examples"]],
                        "col": [e["col"] for e in first["examples"]]})
    ex = d["examples"]
    exact = d["exact"]["wins_from_mixed_starts"]
    return {
        "mixed_states": d["mixed_states"], "games_per_start": d["games_per_start"],
        "support_sizes": d["exact"]["support_sizes"],
        "exact": {"vs_random": exact["random"], "vs_nash": exact["nash"],
                  "vs_best_response": exact["br"]},
        "learners": learners,
        "policy_outputs": {
            "actions": ["U", "D", "L", "R"], "step": 0,
            "states": [{"id": f"mixed{k}", "label": f"Mixed state {k + 1} ({_kind(e['row'])})",
                        "state": e["state"]} for k, e in enumerate(ex)],
            "exact": {"row": [e["row"] for e in ex], "col": [e["col"] for e in ex]},
            "learners": outputs,
        },
    }


def main() -> None:
    out = ROOT / "site" / "src" / "pgResults.json"
    out.write_text(json.dumps(build(), indent=1) + "\n")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
