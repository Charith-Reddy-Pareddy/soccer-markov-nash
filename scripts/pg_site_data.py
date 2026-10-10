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
            "entropy": larger_entropy()}


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
    return {"mean": _avg(br), "runs": br, "tie_nash_mean": _avg(tie), "tie_nash_runs": tie,
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
