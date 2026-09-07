"""Spatial features of a soccer state, in the carrier's frame of reference.

Everything is measured relative to *who has the ball* and which way that player
attacks, so the features are invariant under the board's mirror symmetry. They
are used to ask whether the pure/mixed classification of a stage game can be
predicted from geometry alone.
"""

from __future__ import annotations

from soccer_nash.game import SoccerGame, State


def features(game: SoccerGame, state: State) -> dict:
    x0, y0, x1, y1, b = state
    (cx, cy), (dx, dy) = ((x0, y0), (x1, y1)) if b == 0 else ((x1, y1), (x0, y0))
    forward = 1 if b == 0 else -1  # carrier's attacking x-direction
    goal_x = game.width - 1 if b == 0 else 0

    dx_rel = forward * (dx - cx)  # >0: defender is ahead of the carrier
    dy_rel = dy - cy
    player_dist = abs(cx - dx) + abs(cy - dy)
    forward_cell = (cx + forward, cy)
    intercept_dist = abs(dx - forward_cell[0]) + abs(dy - forward_cell[1])

    return {
        "player_dist": player_dist,
        "adjacent": int(player_dist == 1),
        "same_row": int(cy == dy),
        "same_col": int(cx == dx),
        "dx_rel": dx_rel,
        "abs_dy_rel": abs(dy_rel),
        "defender_ahead": int(dx_rel > 0),
        "defender_ahead_same_row": int(dx_rel == 1 and cy == dy),
        "carrier_goal_dist": abs(goal_x - cx),
        "defender_goal_dist": abs(goal_x - dx),
        "carrier_in_goal_row": int(cy in game.goal_rows),
        "carrier_at_back_wall": int(cy == 0 or cy == game.height - 1),
        "defender_can_intercept": int(intercept_dist <= 1),
        "carrier_can_score_next": int(
            cx == goal_x and cy in game.goal_rows and dx_rel != 0
        ),
    }


FEATURE_NAMES = list(features(SoccerGame(), (0, 2, 6, 2, 0)).keys())
