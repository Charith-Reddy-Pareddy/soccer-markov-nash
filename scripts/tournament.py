"""Reproduce Littman (1994) Table 3: minimax policies are robust, greedy ones
are not.

Littman trained four soccer policies -- MR/MM by minimax-Q, QR/QQ by ordinary
Q-learning -- and evaluated each against a random opponent, a hand-built
opponent, and a *challenger* trained specifically to beat it. His finding: the
minimax policies held ~35-40% of games against their challenger; the Q-learning
(deterministic) policies won **0%**, because "every deterministic offense in
this game has a perfect defense, much like rock, paper, scissors."

This script reproduces the structure with an **exact** solver instead of
learning -- no Monte-Carlo noise. On Littman's board (5x4, two-cell goals, five
actions, random move order, goal-reset scoring, gamma 0.9):

- minimax     -- the exact Nash (minimax-Q fixed point) policy
- greedy/rand -- best response to a uniform-random opponent (Littman's QR)
- greedy/self -- the deterministic maximin-security policy (Littman's QQ-ish)
- hand-built  -- Littman's "simple rules for scoring and blocking" policy
                 (soccer_nash/opponents.py handbuilt_policy)

scored against random / hand-built / a per-policy best-response challenger. The
score is player 0's exact expected discounted goal difference from the kickoff.

    python scripts/tournament.py

Writes `experiments/tournament.csv` and the figure.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.best_response import materialize
from soccer_nash.evaluate import policy_value
from soccer_nash.exploit import best_response_to, onehot_policy, uniform_policy
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.opponents import handbuilt_policy
from soccer_nash.viz import P0, P1, grouped_bars_svg

CSV = pathlib.Path("experiments/tournament.csv")
FIG = pathlib.Path("docs/figures/gallery/tournament.svg")
GAMMA = 0.9
BOARD = {"width": 5, "height": 4, "goal_rows": (1, 2),
         "move_order": "random", "n_actions": 5}


def scripted_policy(game, me):
    """Littman's hand-built policy (deterministic scoring + blocking rules)."""
    return onehot_policy(materialize(game, handbuilt_policy, me), game.n_actions)


def build_policies(game):
    nash = NashQIteration(game, gamma=GAMMA, mode="hybrid", tol=1e-9).run()
    gvr = best_response_to(game, uniform_policy(game), responder=0, gamma=GAMMA)
    gvs = NashQIteration(game, gamma=GAMMA, mode="pure", tol=1e-9).run()
    return {
        "minimax": nash.row_policy,
        "greedy/rand": onehot_policy(gvr.policy, game.n_actions),
        "greedy/self": {s: _argmax_onehot(gvs.row_policy[s]) for s in game.states()},
        "hand-built": scripted_policy(game, 0),
    }


def _argmax_onehot(p):
    v = np.zeros(len(p))
    v[int(np.argmax(p))] = 1.0
    return v


def score(game, row_policy, opponent_kind: str) -> float:
    k = game.initial_state()
    if opponent_kind == "random":
        col = uniform_policy(game)
    elif opponent_kind == "hand-built":
        col = scripted_policy(game, 1)
    else:  # challenger: best response to this policy
        br = best_response_to(game, row_policy, responder=1, gamma=GAMMA)
        return -br.values[k]
    return policy_value(game, row_policy, col, gamma=GAMMA)[k]


def run(scoring: str):
    game = SoccerGame(scoring=scoring, **BOARD)
    policies = build_policies(game)
    opponents = ["random", "hand-built", "challenger"]
    table = {
        name: {opp: score(game, pol, opp) for opp in opponents}
        for name, pol in policies.items()
    }
    return game, table, opponents


def main() -> None:
    print("Littman soccer (5x4, 2-cell goals, 5 actions, random order), "
          f"gamma {GAMMA}\n")

    # sanity: the hand-built opponent should be a competent policy, not a
    # punching bag -- Littman's beat a random opponent 99.5%.
    g = SoccerGame(scoring="win", **BOARD)
    hb = scripted_policy(g, 0)
    v = policy_value(g, hb, uniform_policy(g), gamma=GAMMA)[g.initial_state()]
    print(f"hand-built vs random: P(win) - P(loss) = {v:+.3f}  "
          f"(~{50 + 50 * v:.0f}% wins) -- competent, deterministic\n")

    all_rows = []
    for scoring in ("rate", "win"):
        _game, table, opps = run(scoring)
        unit = "goal diff" if scoring == "rate" else "P(win)-P(loss)"
        print(f"--- scoring = {scoring}   (score = player-0 {unit})")
        print(f"{'policy':>12}  " + "  ".join(f"{o:>11}" for o in opps)
              + f"  {'robustness':>11}")
        for name, row in table.items():
            gap = row["random"] - row["challenger"]
            print(f"{name:>12}  " + "  ".join(f"{row[o]:>+11.3f}" for o in opps)
                  + f"  {gap:>+11.3f}")
            all_rows.append((scoring, name, *(row[o] for o in opps), gap))
        print()

    CSV.parent.mkdir(exist_ok=True)
    with CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scoring", "policy", "vs_random", "vs_hand_built",
                    "vs_challenger", "robustness_gap"])
        w.writerows(all_rows)
    print(f"wrote {CSV}")

    # figure: the rate-scoring tournament
    _game, table, opps = run("rate")
    names = list(table)
    palette = [P0, P1, "var(--ember, #a94e18)", "var(--ink-faint, #77817a)"]
    series = [
        (name, palette[i % len(palette)], [table[name][o] for o in opps])
        for i, name in enumerate(names)
    ]
    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(grouped_bars_svg(
        opps, series, y_label="player-0 goal difference",
        title="Minimax holds up; greedy collapses to its challenger",
    ))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
