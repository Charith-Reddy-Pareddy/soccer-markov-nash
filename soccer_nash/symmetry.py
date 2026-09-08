"""Mirror symmetry of the soccer game.

Reflecting the board left-right and swapping the two players maps the game onto
itself: player 0 (attacking right) becomes player 1 (attacking left, on the
flipped board) and vice versa. Because the reward is zero-sum, this is an
*anti*-symmetry -- ``V(mirror(s)) = -V(s)``.

Every non-terminal state has a distinct mirror image (``b`` flips, so nothing is
self-mirror), so the 2380 states split into exactly 1190 mirror pairs. Solving
one representative per pair and reconstructing the other halves the work.
"""

from __future__ import annotations

import numpy as np

from soccer_nash.game import Action, SoccerGame, State

# Left-right board flip swaps L and R; U and D are unchanged.
_FLIP_ACTION = {
    Action.U: Action.U,
    Action.D: Action.D,
    Action.L: Action.R,
    Action.R: Action.L,
    Action.STAND: Action.STAND,
}
_FLIP_INDEX = np.array([0, 1, 3, 2])  # applied to a length-4 action vector


def mirror_state(state: State, width: int) -> State:
    """Reflect the board and swap the players."""
    x0, y0, x1, y1, b = state
    if x0 == -1:  # terminal
        return (-1, -1, -1, -1, 1 - b)
    w = width - 1
    return (w - x1, y1, w - x0, y0, 1 - b)


def flip_action(a: int) -> int:
    return int(_FLIP_ACTION[Action(a)])


def flip_distribution(dist: np.ndarray) -> np.ndarray:
    """Mirror a length-4 action distribution (swap L and R mass)."""
    return np.asarray(dist, dtype=float)[_FLIP_INDEX]


def canonical_pairs(game: SoccerGame) -> tuple[list[State], dict[State, State]]:
    """``(representatives, image_of)`` -- one state per mirror pair, plus a map
    from every state to its representative."""
    reps: list[State] = []
    image_of: dict[State, State] = {}
    seen: set[State] = set()
    for s in game.states():
        if s in seen:
            continue
        m = mirror_state(s, game.width)
        seen.add(s)
        seen.add(m)
        reps.append(s)
        image_of[s] = s
        image_of[m] = s
    return reps, image_of
