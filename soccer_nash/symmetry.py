"""Mirror symmetry of the soccer game.

Reflecting the board left-right and swapping the two players maps the game onto
itself: player 0 (attacking right) becomes player 1 (attacking left, on the
flipped board) and vice versa. Because the reward is zero-sum, this is an
*anti*-symmetry -- ``V(mirror(s)) = -V(s)``.

Every non-terminal state has a distinct mirror image (``b`` flips, so nothing is
self-mirror), so the 2380 states split into exactly 1190 mirror pairs. Solving
one representative per pair and reconstructing the other halves the work.

The symmetry is not only a value property: the equilibrium *policies* are
equivariant too -- player 0's strategy at ``s`` equals player 1's strategy at
``mirror(s)`` with L and R swapped (:func:`verify_policy_equivariance`).
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
_FLIP_INDEX = np.array([0, 1, 3, 2, 4])  # swap L/R in an action vector (len 4 or 5)


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
    """Mirror an action distribution (swap L and R mass); length 4 or 5."""
    dist = np.asarray(dist, dtype=float)
    return dist[_FLIP_INDEX[: len(dist)]]


def verify_policy_equivariance(
    game: SoccerGame,
    row_policy: dict[State, np.ndarray],
    col_policy: dict[State, np.ndarray],
    tol: float = 1e-6,
) -> float:
    """Max deviation from ``row_policy[s] == flip(col_policy[mirror(s)])`` over
    all states -- 0 means the equilibrium policies respect the mirror symmetry,
    not just the values.

    Exact (0) where the equilibrium is **unique** -- the random game's 56 mixed
    states and every strict pure saddle. Where the stage game has *several*
    equilibria (the deterministic game's degenerate/tied saddles) the *set* of
    equilibria is still mirror-symmetric, but the single representative each
    solver picks by ``argmax`` tie-break need not be, so this can be as large as
    1. ``tol`` is the reporting threshold.
    """
    worst = 0.0
    for s in game.states():
        m = mirror_state(s, game.width)
        want = flip_distribution(col_policy[m])
        worst = max(worst, float(np.abs(np.asarray(row_policy[s]) - want).max()))
    return worst if worst > tol else 0.0


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
