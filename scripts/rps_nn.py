"""Rock-paper-scissors with neural-network players, standard vs fictitious play vs
fictitious play averaging pure (argmax) policies. Two settings: the soccer runs' (Adam, learning
rate 1e-3, entropy bonus 0.01) and a larger entropy bonus (0.2, plain gradient steps at 0.1);
a snapshot every 5 iterations.

    python scripts/rps_nn.py     # experiments/rps_nn.json
"""

from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash import rps_nn

EXP = pathlib.Path(__file__).resolve().parent.parent / "experiments"
ITERATIONS, SEEDS, BATCH = 3000, 3, 64
CONFIGS = {
    "soccer settings": {"lr": 1e-3, "entropy": 0.01, "sgd": False},
    "larger entropy bonus": {"lr": 0.1, "entropy": 0.2, "sgd": True},
}


def main() -> None:
    out = {"iterations": ITERATIONS, "batch": BATCH, "seeds": SEEDS, "snapshot_every": 5,
           "configs": {}}
    for cname, cfg in CONFIGS.items():
        out["configs"][cname] = {**cfg, "modes": {}}
        for mode in rps_nn.MODES:
            runs = [rps_nn.run(mode, ITERATIONS, BATCH, cfg["lr"], 5, seed, cfg["entropy"],
                               cfg["sgd"]) for seed in range(SEEDS)]
            res = {
                "current": runs[0]["current"], "aggregate": runs[0]["aggregate"],
                "final_exploitability": [
                    round(st.mean(r["exploitability"][-300:]), 4) for r in runs],
                "start_exploitability": [
                    round(st.mean(r["exploitability"][:50]), 4) for r in runs],
            }
            out["configs"][cname]["modes"][mode] = res
            print(cname, mode, res["final_exploitability"], flush=True)
    (EXP / "rps_nn.json").write_text(json.dumps(out) + "\n")
    print("wrote experiments/rps_nn.json")


if __name__ == "__main__":
    main()
