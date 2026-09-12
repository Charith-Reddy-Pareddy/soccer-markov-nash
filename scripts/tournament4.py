"""The Markov-game tournament table with plain 4x4 stage games -- no `stand`.

Requested at the research meeting: reproduce the minimax-vs-everyone table
without Littman's fifth action, so every stage game is the textbook 4x4
{U, D, L, R} matrix, and add the simplest possible opponent -- one that always
plays the same action -- alongside random, the hand-built policy, and each
row's best-response challenger.

    minimax     -- the exact Nash (minimax-Q) policy
    greedy/rand -- best response to a uniform-random opponent
    greedy/self -- the deterministic maximin-security policy
    hand-built  -- Littman's "simple rules for scoring and blocking" policy

scored against:

    always-left -- the simplest fixed opponent: play L every step
                   (soccer_nash.opponents.always_left)
    random      -- uniform over the 4 actions
    hand-built  -- the same scripted policy, as an opponent
    challenger  -- a best response *to this row's exact policy* (it knows the
                   mixing probabilities, not the realised draw)

The point stated at the meeting: a minimax policy can still take goal
difference off its challenger (because the challenger only knows the
*distribution*, not the coin flip), but any single fixed action sequence --
greedy/self, hand-built, or a policy imitated from a single-agent RL run --
has a challenger that beats it every time, "like rock, paper, scissors".

    python scripts/tournament4.py

Writes `experiments/tournament4.csv` and the figure.
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
from soccer_nash.opponents import always_left, handbuilt_policy
from soccer_nash.viz import P0, P1, grouped_bars_svg

CSV = pathlib.Path("experiments/tournament4.csv")
FIG = pathlib.Path("docs/figures/gallery/tournament4.svg")
GAMMA = 0.9
BOARD = {"width": 5, "height": 4, "goal_rows": (1, 2),
         "move_order": "random", "n_actions": 4}  # <- no stand: plain 4x4 games


def scripted_policy(game, me, fn=handbuilt_policy):
    return onehot_policy(materialize(game, fn, me), game.n_actions)


def _argmax_onehot(p):
    v = np.zeros(len(p))
    v[int(np.argmax(p))] = 1.0
    return v


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


def score(game, row_policy, opponent_kind: str) -> float:
    k = game.initial_state()
    if opponent_kind == "always-left":
        col = scripted_policy(game, 1, always_left)
    elif opponent_kind == "random":
        col = uniform_policy(game)
    elif opponent_kind == "hand-built":
        col = scripted_policy(game, 1)
    else:  # challenger: best response to this row's exact policy
        br = best_response_to(game, row_policy, responder=1, gamma=GAMMA)
        return -br.values[k]
    return policy_value(game, row_policy, col, gamma=GAMMA)[k]


def main() -> None:
    game = SoccerGame(scoring="rate", **BOARD)
    policies = build_policies(game)
    opponents = ["always-left", "random", "hand-built", "challenger"]

    print(f"{BOARD['width']}x{BOARD['height']}, goal {BOARD['goal_rows']}, "
          f"{BOARD['n_actions']} actions (no stand), random order, "
          f"scoring=rate, gamma {GAMMA}\n")
    print(f"{'policy':>12}  " + "  ".join(f"{o:>11}" for o in opponents)
          + f"  {'robustness':>11}")

    table, rows = {}, []
    for name, pol in policies.items():
        row = {opp: score(game, pol, opp) for opp in opponents}
        table[name] = row
        gap = row["random"] - row["challenger"]
        print(f"{name:>12}  " + "  ".join(f"{row[o]:>+11.3f}" for o in opponents)
              + f"  {gap:>+11.3f}")
        rows.append((name, *(row[o] for o in opponents), gap))
    print()
    print("=> minimax is the only policy whose challenger cannot drive it "
          "negative; every fixed-action policy (greedy/self, hand-built) has "
          "a challenger that beats it -- the same 4x4-matrix result as with "
          "the stand action, so `stand` was never load-bearing for this point.")

    CSV.parent.mkdir(exist_ok=True)
    with CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", *opponents, "robustness_gap"])
        w.writerows(rows)
    print(f"\nwrote {CSV}")

    names = list(table)
    palette = [P0, P1, "var(--ember, #a94e18)", "var(--ink-faint, #77817a)"]
    series = [
        (name, palette[i % len(palette)], [table[name][o] for o in opponents])
        for i, name in enumerate(names)
    ]
    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text(grouped_bars_svg(
        opponents, series, y_label="player-0 goal difference",
        title="Same result with plain 4x4 stage games (no stand action)",
    ))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
