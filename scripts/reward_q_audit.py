"""Reproduce reward differences and measure learned Q errors, without changing defaults.

Optional --reference points to JSON extracted from Jae's public viewer/data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration

CASES = [
    (4, 3, 5, 3, 0),
    (0, 0, 0, 2, 0),
    (3, 4, 4, 4, 0),
    (6, 1, 4, 1, 0),
    (0, 0, 0, 1, 1),
    (0, 1, 0, 3, 0),
    (1, 0, 0, 1, 0),
    (0, 0, 2, 0, 0),
]
SCREENSHOT = np.array(
    [
        [0.2, 0.2, -0.15, 0.2],
        [0.2, 0.277113, 0.748415, 0.245],
        [0.15, 0.195, 0.15, 0.195],
        [-0.15, 0.299050, -0.705769, 0.25],
    ]
)


class PossessionProgressGame(SoccerGame):
    """Jae's public viewer reward, 2026-09-28; opt-in comparison only.

    https://jwsong118.github.io/soccer-mpe/soccer-viewer.js
    Nonterminal reward uses NEXT state. Goals remain +/-1 without bonuses.
    This differs from StepProgressionBonus's pre-state, unconditional x0.
    """

    def transitions(self, state, a0, a1):
        outcomes = super().transitions(state, a0, a1)
        result = []
        for prob, ns, reward in outcomes:
            if not self.is_terminal(ns):
                r = (
                    0.005 + 0.005 * ns[0]
                    if ns[4] == 0
                    else -(0.005 + 0.005 * (self.width - 1 - ns[2]))
                )
                reward = (r, -r)
            result.append((prob, ns, reward))
        return result


def audit(epochs=400, iterations=1000, seeds=(0, 1), reference=None):
    import torch

    from soccer_nash.nash_dqn import train_nash_dqn
    from soccer_nash.policy_gradient import train_reinforce_selfplay

    torch.set_num_threads(1)
    report = {
        "gamma": 0.9,
        "dqn_epochs": epochs,
        "pg_iterations": iterations,
        "hidden": 64,
        "pg_rollout_len": 100,
        "seeds": list(seeds),
        "models": {},
    }
    for name, cls in [("sparse", SoccerGame), ("possession_progress", PossessionProgressGame)]:
        game = cls(width=7, height=5, goal_rows=(1, 2, 3), move_order="deterministic")
        states = list(game.states())
        solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-11)
        exact = solver.run_exact()
        matrices = np.array([solver._matrix(s, exact.values) for s in states])
        values = np.array([exact.values[s] for s in states])
        # Full equilibrium certificate including mixed policies and deviations.
        lo = np.array([(exact.row_policy[s] @ m).min() for s, m in zip(states, matrices)])
        hi = np.array([(m @ exact.col_policy[s]).max() for s, m in zip(states, matrices)])
        entry = {
            "states": len(states),
            "all_zero_matrices": int(np.all(matrices == 0, axis=(1, 2)).sum()),
            "zero_value_states": int((values == 0).sum()),
            "max_bellman_residual": float(np.max(np.abs(lo - values))),
            "max_equilibrium_gap": float(np.max(hi - lo)),
            "cases": [
                {
                    "state": s,
                    "value": exact.values[s],
                    "Q": solver._matrix(s, exact.values).tolist(),
                }
                for s in CASES
            ],
            "training": [],
        }
        if name == "possession_progress":
            entry["screenshot_max_error"] = float(
                np.max(np.abs(solver._matrix(CASES[2], exact.values) - SCREENSHOT))
            )
            if reference:
                ref = json.loads(Path(reference).read_text())
                entry["reference_value_max_error"] = float(
                    max(
                        abs(exact.values[tuple(s)] - v)
                        for s, v in zip(ref["states"], ref["values"])
                    )
                )
                mismatches = 0
                for s, pairs in zip(ref["states"], ref["transitions"]):
                    from soccer_nash.game import MOVE_ACTIONS

                    for k, target in enumerate(pairs):
                        _, ns, reward = game.transitions(
                            tuple(s), MOVE_ACTIONS[k // 4], MOVE_ACTIONS[k % 4]
                        )[0]
                        expected_ns = None if game.is_terminal(ns) else list(ns)
                        mismatches += (
                            expected_ns != target["ns"] or abs(reward[0] - target["reward"]) > 1e-12
                        )
                entry["reference_transition_reward_mismatches"] = mismatches
                ref_values = dict(zip(map(tuple, ref["states"]), ref["values"]))
                q_error = 0.0
                for state, pairs in zip(ref["states"], ref["transitions"]):
                    matrix = solver._matrix(tuple(state), exact.values)
                    for k, target in enumerate(pairs):
                        value = target["reward"] + (
                            0.9 * ref_values[tuple(target["ns"])] if target["ns"] else 0.0
                        )
                        q_error = max(q_error, abs(matrix[k // 4, k % 4] - value))
                certificate = {
                    "transition_reward_pairs": len(states) * 16,
                    "mismatches": mismatches,
                    "value_max_error": entry["reference_value_max_error"],
                    "Q_max_error": q_error,
                    "max_bellman_residual": float(
                        max(np.max(np.abs(lo - values)), np.max(np.abs(hi - values)))
                    ),
                    "max_equilibrium_gap": float(np.max(hi - lo)),
                    "mixed_states": len(exact.no_saddle_states),
                    "screenshot_max_error": entry["screenshot_max_error"],
                }
                Path("experiments/reward_reference_audit.json").write_text(
                    json.dumps(certificate, indent=2) + "\n"
                )
        print(name, {k: v for k, v in entry.items() if k not in ("cases", "training")}, flush=True)
        for seed in seeds:
            dqn = train_nash_dqn(game, epochs=epochs, seed=seed).net
            dq = dqn.predict(np.array(states, float)).reshape(-1, 4, 4)
            pg = train_reinforce_selfplay(game, iterations=iterations, seed=seed)
            p = {s: pg.net0.policy(s).astype(float) for s in states}
            q = {s: pg.net1.policy(s).astype(float) for s in states}
            for s in states:
                p[s] /= p[s].sum()
                q[s] /= q[s].sum()
            pv = solver.solve_exact(replace(exact, row_policy=p, col_policy=q))
            pq = np.array([solver._matrix(s, pv) for s in states])
            run = {
                "seed": seed,
                "dqn_Q_mae": float(np.abs(dq - matrices).mean()),
                "dqn_Q_max_error": float(np.abs(dq - matrices).max()),
                "pg_policy_Q_mae": float(np.abs(pq - matrices).mean()),
                "pg_policy_Q_max_error": float(np.abs(pq - matrices).max()),
                "cases": [
                    {
                        "state": s,
                        "dqn_Q": dq[states.index(s)].tolist(),
                        "pg_policy_Q": pq[states.index(s)].tolist(),
                        "pg_p0": p[s].tolist(),
                        "pg_p1": q[s].tolist(),
                    }
                    for s in CASES
                ],
            }
            entry["training"].append(run)
            report["models"][name] = entry
            Path("experiments/reward_q_audit.json").write_text(json.dumps(report, indent=2) + "\n")
            print(name, seed, {k: v for k, v in run.items() if k != "cases"}, flush=True)
        report["models"][name] = entry
        Path("experiments/reward_q_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    audit(args.epochs, args.iterations, reference=args.reference)
