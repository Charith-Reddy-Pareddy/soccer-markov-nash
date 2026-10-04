"""Self-play PPO with an angle-radius policy on the provisional dog-and-sheep game.

    python scripts/dog_game.py --seeds 2     # writes experiments/dog_game.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash import dog_game as dg

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "dog_game.csv"


def greedy(policy):
    return lambda x: policy.act(x, greedy=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=300)
    args = ap.parse_args()

    rows = []
    for seed in range(args.seeds):
        dog, sheep = dg.train_ppo_selfplay(iterations=args.iterations, seed=seed)
        with torch.no_grad():
            matchups = {
                "random dog vs random sheep": (dg.random_act, dg.random_act),
                "trained dog vs random sheep": (greedy(dog), dg.random_act),
                "random dog vs trained sheep": (dg.random_act, greedy(sheep)),
                "trained dog vs trained sheep": (greedy(dog), greedy(sheep)),
                "straight-line dog vs trained sheep": (dg.greedy_dog_act, greedy(sheep)),
            }
            for name, (d, s) in matchups.items():
                r = dg.play(d, s, n_games=500, seed=seed)
                r = {k: round(v, 3) for k, v in r.items()}
                rows.append({"seed": seed, "matchup": name, **r})
                print(seed, name, r["capture_rate"], r["mean_capture_step"], flush=True)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote experiments/dog_game.csv")


if __name__ == "__main__":
    main()
