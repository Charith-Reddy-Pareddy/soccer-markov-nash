"""Rock-paper-scissors with neural networks: how long each best response is trained.

Fictitious play needs each snapshot to be a good best response to the average so far. With the
soccer settings the networks are trained for 5 iterations between snapshots; here the phase is
5, 50 or 200 iterations, for the average of softmax snapshots and of argmax (pure) snapshots.

    python scripts/rps_nn_phases.py     # experiments/rps_nn_phases.json
"""

from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash import rps_nn

EXP = pathlib.Path(__file__).resolve().parent.parent / "experiments"
ITERATIONS, SEEDS, BATCH, LR, ENTROPY = 8000, 3, 64, 1e-3, 0.01
PHASES = (5, 50, 200)


def main() -> None:
    out = {"iterations": ITERATIONS, "batch": BATCH, "seeds": SEEDS, "lr": LR, "entropy": ENTROPY,
           "phases": list(PHASES), "runs": {}}
    for phase in PHASES:
        for mode in ("fictitious", "fictitious_argmax"):
            runs = [rps_nn.run(mode, ITERATIONS, BATCH, LR, phase, seed, ENTROPY)
                    for seed in range(SEEDS)]
            final = [round(st.mean(r["exploitability"][-300:]), 4) for r in runs]
            out["runs"][f"{phase}/{mode}"] = {
                "phase": phase, "mode": mode, "final_exploitability": final,
                "aggregate": runs[0]["aggregate"][::10], "current": runs[0]["current"][::10]}
            print(phase, mode, final, flush=True)
    (EXP / "rps_nn_phases.json").write_text(json.dumps(out) + "\n")
    print("wrote experiments/rps_nn_phases.json")


if __name__ == "__main__":
    main()
