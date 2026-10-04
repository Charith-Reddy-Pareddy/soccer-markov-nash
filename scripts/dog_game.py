"""Self-play PPO with an angle-radius policy on the provisional dog-and-sheep game.

    python scripts/dog_game.py --seeds 2         # PPO, writes experiments/dog_game.csv
    python scripts/dog_game.py dqn --seeds 2     # 10-angle DQN, writes dog_game_dqn.csv
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


def run_dqn(seeds: int, iterations: int) -> None:
    rows = []
    for seed in range(seeds):
        q = dg.train_angle_dqn(iterations=iterations, seed=seed)
        with torch.no_grad():
            dog = dg.angle_dqn_act(q, 10)
            for name, sheep in (("fleeing", dg.fleeing_sheep_act), ("random", dg.random_act)):
                r = {k: round(v, 3) for k, v in dg.play(dog, sheep, 500, seed).items()}
                rows.append({"seed": seed, "matchup": f"10-angle DQN dog vs {name} sheep", **r})
                print(rows[-1], flush=True)
        r = {k: round(v, 3) for k, v in dg.play(
            dg.greedy_dog_act, dg.fleeing_sheep_act, 500, seed).items()}
        rows.append({"seed": seed, "matchup": "straight-line dog vs fleeing sheep", **r})
    path = OUT.with_name("dog_game_dqn.csv")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote experiments/{path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", nargs="?", choices=["ppo", "dqn"], default="ppo")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=300)
    args = ap.parse_args()
    if args.which == "dqn":
        return run_dqn(args.seeds, args.iterations)

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
