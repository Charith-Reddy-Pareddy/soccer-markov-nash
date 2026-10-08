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
            "learners": learners, "longer": longer, "fp_br": fp_br}


def main() -> None:
    out = ROOT / "site" / "src" / "pgResults.json"
    out.write_text(json.dumps(build(), indent=1) + "\n")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
