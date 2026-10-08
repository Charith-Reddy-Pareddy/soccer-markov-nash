"""Action probabilities of the trained policy-gradient learners at fixed states.

    python scripts/pg_policy_outputs.py --algo a2c --mode selfplay   # one learner
    python scripts/pg_policy_outputs.py --merge                      # combine the parts

Each run trains one learner exactly as ``pg_finite.py`` does (same seed, so the
same network as that run's row) and saves the probabilities of U, D, L, R for both
players at the start of the game, next to the exact solver's action.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash import finite_horizon as fh
from soccer_nash import pg_finite as pf
from soccer_nash.game import A10SoccerGame
from soccer_nash.nash_q import NashQIteration

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
ACTIONS = ["U", "D", "L", "R"]
STATES = [
    ("kickoff", "Kickoff", (0, 1, 6, 3, 0)),
    ("contest", "Players side by side in a goal row", (3, 2, 4, 2, 0)),
    ("attack", "Carrier near the goal, defender far away", (5, 1, 6, 3, 0)),
    ("defend", "The right player holds the ball, players stacked", (3, 2, 3, 3, 1)),
]
LABELS = {"reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
MODES = {"selfplay": "standard", "fictitious": "fictitious play"}


def merge() -> None:
    parts = [json.loads(pathlib.Path(p).read_text())
             for p in sorted(glob.glob(str(EXP / "pg_policy_outputs_*_*.json")))]
    if not parts:
        raise SystemExit("no part files to merge")
    out = {"actions": ACTIONS, "step": 0, "exact": parts[0]["exact"],
           "states": [{"id": i, "label": lbl, "state": list(s)} for i, lbl, s in STATES],
           "learners": [p["learner"] for p in parts]}
    order = {(a, m): k for k, (a, m) in enumerate(
        (a, m) for a in LABELS for m in MODES)}
    for lr in (p["learner"] for p in parts):
        lr["label"] = f"{LABELS[lr['algo']]}, {MODES[lr['mode']]}"
    out["learners"].sort(key=lambda lr: order[(lr["algo"], lr["mode"])])
    text = json.dumps(out, indent=1) + "\n"
    (EXP / "pg_policy_outputs.json").write_text(text)
    (ROOT / "site" / "src" / "pgPolicies.json").write_text(text)
    print(f"merged {len(parts)} learners")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", choices=list(LABELS))
    ap.add_argument("--mode", choices=list(MODES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge()

    game = A10SoccerGame()
    horizon = game.max_steps
    states = [s for _, _, s in STATES]
    solver = NashQIteration(game, gamma=a.gamma, mode="hybrid", tol=1e-10)
    _, row_t, col_t = fh.solve_finite_horizon(solver, a.gamma, horizon)
    exact = {"row": [row_t[0][s].round(4).tolist() for s in states],
             "col": [col_t[0][s].round(4).tolist() for s in states]}
    tr = pf.train(game, a.algo, a.mode, a.gamma, horizon, a.iterations, 64, seed=a.seed)
    learner = {"algo": a.algo, "mode": a.mode, "seed": a.seed, "iterations": a.iterations,
               "label": f"{LABELS[a.algo]}, {MODES[a.mode]}",
               "row": tr.pol0(0, states).round(4).tolist(),
               "col": tr.pol1(0, states).round(4).tolist()}
    path = EXP / f"pg_policy_outputs_{a.algo}_{a.mode}.json"
    path.write_text(json.dumps({"exact": exact, "learner": learner}, indent=1) + "\n")
    print("wrote", path.name, learner["row"][0])


if __name__ == "__main__":
    main()
