"""Minimax vs. always-left vs. random vs. best response -- the full 4x4
policy-vs-policy matrix, exactly, on the same canonical board every case in
docs/positions.md is drawn from. Player 0's policy is the row, player 1's the
column; every cell is player 0's exact expected discounted goal difference
from kickoff (``scoring="rate"``, matching scripts/tournament4.py).

"Best response" is not a fixed policy -- it best-responds to whatever the
*other* side is playing, exactly the resolution rule
``site/src/explorer/helpers.js``'s ``resolvePolicies`` uses for the live
Simulate panel, so this table and the live site's numbers agree. Where
*both* sides are "best response" there is no order to resolve first, so both
fall back to the minimax policy -- which is, not coincidentally, the actual
fixed point of "best-respond to a best-response".

    python scripts/positions_policy_matrix.py

Prints the matrix; writes experiments/positions_policy_matrix.csv.
"""

from __future__ import annotations

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from soccer_nash.evaluate import policy_value
from soccer_nash.exploit import best_response_to, onehot_policy, uniform_policy
from soccer_nash.game import SoccerGame
from soccer_nash.nash_q import NashQIteration
from soccer_nash.opponents import always_left

CSV = pathlib.Path("experiments/positions_policy_matrix.csv")
GAMMA = 0.9
BOARD = {"width": 7, "height": 5, "goal_rows": (1, 2, 3), "move_order": "random"}
POLICIES = ["minimax", "always-left", "random", "best-response"]


def build_fixed_policies(game):
    """The three policies that don't depend on the opponent."""
    nash = NashQIteration(game, gamma=GAMMA, mode="hybrid", tol=1e-10).run()
    return {
        "minimax": (nash.row_policy, nash.col_policy),
        "always-left": (
            onehot_policy({s: always_left(game, s, 0) for s in game.states()}),
            onehot_policy({s: always_left(game, s, 1) for s in game.states()}),
        ),
        "random": (uniform_policy(game), uniform_policy(game)),
    }


def cell_value(game, row_name, col_name, fixed):
    """Player 0's exact value with row playing ``row_name`` against column
    playing ``col_name``. Resolves "best-response" against whatever the
    other side turns out to be; both "best-response" falls back to minimax."""
    row_is_br, col_is_br = row_name == "best-response", col_name == "best-response"
    if row_is_br and col_is_br:
        row_pol, col_pol = fixed["minimax"]
        return policy_value(game, row_pol, col_pol, gamma=GAMMA)[game.initial_state()]
    s0 = game.initial_state()
    if col_is_br:
        row_pol = fixed[row_name][0]
        # best_response_to returns the *responder's* (player 1's) own value.
        return -best_response_to(game, row_pol, responder=1, gamma=GAMMA).values[s0]
    if row_is_br:
        col_pol = fixed[col_name][1]
        return best_response_to(game, col_pol, responder=0, gamma=GAMMA).values[s0]
    row_pol, col_pol = fixed[row_name][0], fixed[col_name][1]
    return policy_value(game, row_pol, col_pol, gamma=GAMMA)[s0]


def main() -> None:
    game = SoccerGame(scoring="rate", **BOARD)
    fixed = build_fixed_policies(game)

    print(f"{BOARD['width']}x{BOARD['height']}, goal {BOARD['goal_rows']}, "
          f"random move order, scoring=rate, gamma {GAMMA}\n")
    print("player 0's goal difference, row = player 0's policy, col = player 1's\n")
    print(f"{'':>15}" + "".join(f"{c:>16}" for c in POLICIES))

    rows = []
    for r in POLICIES:
        vals = [cell_value(game, r, c, fixed) for c in POLICIES]
        print(f"{r:>15}" + "".join(f"{v:>+16.4f}" for v in vals))
        rows.append((r, *vals))

    CSV.parent.mkdir(exist_ok=True)
    with CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", *POLICIES])
        w.writerows(rows)
    print(f"\nwrote {CSV}")


if __name__ == "__main__":
    main()
