"""Exact (noise-free) gradient play on both boards: a table of logits against a shared network,
and plain against extragradient steps on the random board. Appends one row per checkpoint to
experiments/pg_exact_play.csv and skips runs already there.

    python scripts/pg_exact_play.py                       # every run
    python scripts/pg_exact_play.py --only random-eg      # one group (a10, random, random-eg)
    python scripts/pg_exact_play.py --shard 0/4           # every fourth run, from the first
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from soccer_nash.exact_play import play
from soccer_nash.game import A10SoccerGame, SoccerGame

OUT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "pg_exact_play.csv"
FIELDS = ["board", "policy", "method", "lr", "seed", "init", "steps",
          "step", "gap_mean", "gap_max", "gap_kickoff"]

# (board, policy, method, step size, seed, initial logit scale, steps)
RUNS = (
    [("a10", "table", "adam", 0.05, 0, 0.0, 2000)]
    + [("a10", "network", "adam", 0.003, s, 0.0, 2000) for s in range(5)]
    + [("random", "table", "adam", 0.05, 0, 0.0, 2000)]
    + [("random", "network", "adam", 0.003, s, 0.0, 2000) for s in range(5)]
    + [("random", "table", m, lr, s, 0.1, 6000)
       for m, lr in (("sgd", 0.5), ("eg", 0.25), ("eg", 0.5)) for s in range(3)]
)
GROUPS = {"a10": lambda r: r[0] == "a10",
          "random": lambda r: r[0] == "random" and r[6] == 2000,
          "random-eg": lambda r: r[6] == 6000}


def done() -> set:
    if not OUT.exists():
        return set()
    with OUT.open() as f:
        return {(r["board"], r["policy"], r["method"], float(r["lr"]), int(r["seed"]),
                 float(r["init"]), int(r["steps"])) for r in csv.DictReader(f)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(GROUPS))
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    torch.set_num_threads(1)
    first, every = (int(x) for x in args.shard.split("/"))
    runs = [r for r in RUNS if args.only is None or GROUPS[args.only](r)][first::every]
    finished = done()
    for run in runs:
        if run in finished:
            continue
        board, policy, method, lr, seed, init, steps = run
        game = A10SoccerGame() if board == "a10" else SoccerGame(move_order="random")
        rows = play(game, policy, method, steps=steps, lr=lr, seed=seed, init=init,
                    every=100 if steps == 2000 else 250)
        new = not OUT.exists()
        with OUT.open("a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(FIELDS)
            writer.writerows([list(run) + list(r) for r in rows])
        _, mean, biggest, kickoff = rows[-1]
        print(*run, f"final gap mean {mean:.4f} max {biggest:.4f} kickoff {kickoff:.4f}",
              flush=True)
