"""Train one representative DQN net and one PG self-play run per board
configuration, and export their per-state predictions for the interactive
explorer's neural cross-check panel.

This is *not* the definitive multi-seed research claim -- docs/neural.md and
experiments/*.csv own that, with proper seed averaging and error bars. This
is a single, clearly-labelled representative run per board (seed 0), cheap
enough to regenerate on demand, so a reader looking at *any* state in the
live explorer -- not just the handful already written up in a case study --
can see what an independent learned method predicts there, next to the
exact solve.

DQN is the fitted-Q "from zero" baseline (soccer_nash.nash_dqn
.train_nash_dqn): pure TD bootstrap from a random init, the same starting
point docs/neural.md reports as the primary, least-favourable-to-itself
comparison (not "fit to exact", which is trained by regressing directly onto
the exact answer and so isn't an independent check of anything). PG is
self-play REINFORCE (soccer_nash.policy_gradient.train_reinforce_selfplay):
genuinely on-policy, so its predictions at a state rarely reached from
kickoff under its own policy are expected to be unreliable -- that is
reported as data (a real limitation of self-play), not hidden.

    python scripts/explorer_neural_data.py

Writes docs/data/explorer_neural.json, fetched only when the explorer's
"Neural cross-check" panel is opened (not on every page load).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from explorer_data import BOARDS

from soccer_nash.game import SoccerGame
from soccer_nash.matrix_games import pure_bounds, solve_zero_sum
from soccer_nash.nash_dqn import _SADDLE_TOL, train_nash_dqn
from soccer_nash.policy_gradient import train_reinforce_selfplay

OUT = pathlib.Path("docs/data/explorer_neural.json")
DQN_EPOCHS = 400
PG_ITERATIONS = 1000
SEED = 0


def r4(x: float) -> float:
    return round(float(x), 4)


def extract_policy(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same extraction soccer_nash.nash_dqn.compare_to_exact uses: the pure
    fast path when the predicted matrix has (near) no gap, the LP otherwise
    -- so a policy read off a noisy predicted matrix is handled the same way
    the exact solver's own matrices are."""
    lo, hi = pure_bounds(m)
    if hi - lo <= _SADDLE_TOL:
        p = np.zeros(4)
        p[int(np.argmax(m.min(axis=1)))] = 1.0
        q = np.zeros(4)
        q[int(np.argmin(m.max(axis=0)))] = 1.0
        return p, q
    _, p, q = solve_zero_sum(m)
    return p, q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pg-only', action='store_true',
        help='Refresh PG while preserving existing DQN predictions',
    )
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    previous = json.loads(OUT.read_text()) if args.pg_only else None
    boards: dict[str, dict] = {}
    for board_id, kw in BOARDS.items():
        print(f"=== {board_id} ===", flush=True)
        g = SoccerGame(**kw)
        states = list(g.states())

        t0 = time.time()
        dqn = None if args.pg_only else train_nash_dqn(
            g, gamma=0.9, hidden=64, epochs=DQN_EPOCHS, seed=SEED,
        )
        if args.pg_only:
            print(f"  preserved DQN predictions ({len(states)} states)", flush=True)
        else:
            print(f"  DQN trained in {time.time() - t0:.1f}s ({len(states)} states)", flush=True)

        t0 = time.time()
        pg = train_reinforce_selfplay(g, gamma=0.9, hidden=64, iterations=PG_ITERATIONS, seed=SEED)
        print(f"  PG trained in {time.time() - t0:.1f}s", flush=True)

        state_data = {}
        for s in states:
            key = ",".join(map(str, s))
            if previous:
                old = previous['boards'][board_id][key]
                Qd, p_dqn, q_dqn = old[:3]
            else:
                Qd = dqn.net.matrix(s)
                p_dqn, q_dqn = extract_policy(Qd)
            p_pg = pg.net0.policy(s)
            q_pg = pg.net1.policy(s)
            state_data[key] = [
                [[r4(v) for v in row] for row in Qd],
                [r4(v) for v in p_dqn], [r4(v) for v in q_dqn],
                [r4(v) for v in p_pg], [r4(v) for v in q_pg],
            ]
        boards[board_id] = state_data
        print(f"  exported {len(state_data)} states", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "meta": {
            "dqn_epochs": DQN_EPOCHS,
            "pg_iterations": PG_ITERATIONS,
            "pg_terminal_boundaries": True,
            "seed": SEED,
            "note": (
                "One representative run per board (seed 0). PG returns stop "
                "at terminal goals. See docs/reward-q-audit.md for the "
                "corrected two-seed comparison and limits of neural validation."
            ),
        },
        "boards": boards,
    }))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
