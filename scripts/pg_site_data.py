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

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"

LEARNERS = [
    ("REINFORCE", "self-play", "reinforce", "selfplay"),
    ("REINFORCE", "fictitious play", "reinforce", "fictitious"),
    ("A2C", "self-play", "a2c", "selfplay"),
    ("A2C", "fictitious play", "a2c", "fictitious"),
    ("PPO", "self-play", "ppo", "selfplay"),
    ("PPO", "fictitious play", "ppo", "fictitious"),
]
NAMES = {"reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}


def read(path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def mean(rows, key) -> float:
    return st.mean(float(r[key]) for r in rows)


def wtl(rows, opp) -> list[float]:
    return [round(mean(rows, f"row_{k}_vs_{opp}"), 6) for k in ("win", "tie", "loss")]


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
                "runs": [round(float(r["exploitability"]), 6) for r in lr],
                "vs_random": wtl(lr, "random"), "vs_nash": wtl(lr, "nash"),
            })
    fp_br = []
    for path in sorted(glob.glob(str(EXP / "pg_fp_br_*.csv"))):
        rs = read(path)
        fp_br.append({
            "label": NAMES[rs[0]["algo"]], "best_response_iterations": int(rs[0]["br_iters"]),
            "checkpoints": [{"round": int(r["round"]),
                             "exploitability": round(float(r["exploitability"]), 6)} for r in rs],
        })
    return {"exact": summary([r for r in main if r["algo"] == "exact"]),
            "learners": learners, "longer": longer, "fp_br": fp_br,
            "random": random_board(), "mixed": mixed_states()}


def random_board() -> dict:
    """The same six learners on the random move-order board (3 seeds each)."""
    rows = [r for p in sorted(glob.glob(str(EXP / "pg_finite_random_*.csv"))) for r in read(p)]
    learners = []
    for label, training, algo, mode in LEARNERS:
        rs = [r for r in rows if r["algo"] == algo and r["mode"] == mode]
        learners.append({"label": label, "training": training, **summary(rs)})
    return {"exact": summary([r for r in rows if r["algo"] == "exact"]), "learners": learners}


def _avg(values) -> float:
    return round(st.mean(values), 6)


def _kind(row) -> str:
    return "three-way mix" if sum(v > 1e-6 for v in row) >= 3 else "two-way mix"


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
