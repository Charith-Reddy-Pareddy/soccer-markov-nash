"""Two-player soccer Markov game (CS 540 A10 geometry).

State: ``(x0, y0, x1, y1, b)`` where ``(x0, y0)`` is player 0's cell,
``(x1, y1)`` is player 1's cell and ``b in {0, 1}`` is the ball carrier.

Terminal states are encoded ``(-1, -1, -1, -1, winner)``.

Dynamics are deterministic and simultaneous-move: both players pick an action
from ``{U, D, L, R}`` and the successor is a pure function of the joint action.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

State = tuple[int, int, int, int, int]


class Action(IntEnum):
    U = 0
    D = 1
    L = 2
    R = 3


# (dx, dy) per action. Up increases y.
_DELTA: dict[Action, tuple[int, int]] = {
    Action.U: (0, 1),
    Action.D: (0, -1),
    Action.L: (-1, 0),
    Action.R: (1, 0),
}

# Joint-action order used by A10 Question 1: UU, UD, UL, UR, DU, ...
JOINT_ACTIONS: list[tuple[Action, Action]] = [
    (a0, a1) for a0 in Action for a1 in Action
]


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


@dataclass(frozen=True)
class SoccerGame:
    width: int = 7
    height: int = 5
    goal_rows: tuple[int, ...] = (1, 2, 3)
    max_steps: int = 100

    # ------------------------------------------------------------------ basics
    def actions(self) -> list[Action]:
        return list(Action)

    def is_terminal(self, state: State) -> bool:
        return state[0] == -1

    def winner(self, state: State) -> int | None:
        return state[4] if self.is_terminal(state) else None

    # ----------------------------------------------------------------- states
    def states(self) -> Iterator[State]:
        """All legal non-terminal states (players never share a cell)."""
        for x0 in range(self.width):
            for y0 in range(self.height):
                for x1 in range(self.width):
                    for y1 in range(self.height):
                        if (x0, y0) == (x1, y1):
                            continue
                        yield (x0, y0, x1, y1, 0)
                        yield (x0, y0, x1, y1, 1)

    def initial_state(self) -> State:
        """Default kickoff: carriers on the centre row at opposite ends."""
        mid = self.height // 2
        return (0, mid, self.width - 1, mid, 0)

    # ------------------------------------------------------------- transition
    def _target(
        self, player: int, pos: tuple[int, int], action: Action, has_ball: bool
    ) -> tuple[tuple[int, int], bool]:
        """Return (clamped target cell, scored?) for one player in isolation."""
        (x, y) = pos
        dx, dy = _DELTA[action]
        rx, ry = x + dx, y + dy

        scored = False
        if has_ball and y in self.goal_rows:
            if player == 0 and rx >= self.width:  # player 0 attacks the right edge
                scored = True
            elif player == 1 and rx < 0:  # player 1 attacks the left edge
                scored = True

        return (_clamp(rx, 0, self.width - 1), _clamp(ry, 0, self.height - 1)), scored

    def step(
        self, state: State, a0: Action, a1: Action
    ) -> tuple[State, tuple[int, int], bool]:
        """Apply a joint action. Returns (next_state, (r0, r1), done)."""
        if self.is_terminal(state):
            raise ValueError("step() called on a terminal state")

        x0, y0, x1, y1, b = state
        p0, p1 = (x0, y0), (x1, y1)

        (t0, s0) = self._target(0, p0, a0, has_ball=(b == 0))
        (t1, s1) = self._target(1, p1, a1, has_ball=(b == 1))

        # Only the carrier can score, so at most one of s0/s1 is true.
        if s0:
            return (-1, -1, -1, -1, 0), (1, -1), True
        if s1:
            return (-1, -1, -1, -1, 1), (-1, 1), True

        if t0 == t1:
            # Both want the same cell: carrier takes it, other keeps its cell
            # and steals the ball. A carrier blocked by a standing opponent
            # cannot advance.
            new_b = 1 - b
            if b == 0:
                n1 = p1
                n0 = t0 if t0 != n1 else p0
            else:
                n0 = p0
                n1 = t1 if t1 != n0 else p1
        elif t0 == p1 and t1 == p0:
            # Swap: players exchange cells and possession flips.
            n0, n1 = p1, p0
            new_b = 1 - b
        else:
            n0, n1 = t0, t1
            new_b = b

        return (n0[0], n0[1], n1[0], n1[1], new_b), (0, 0), False

    def successors(self, state: State) -> list[tuple[State, tuple[int, int]]]:
        """Successor state and reward for every joint action, in A10 order."""
        out: list[tuple[State, tuple[int, int]]] = []
        for a0, a1 in JOINT_ACTIONS:
            nxt, reward, _ = self.step(state, a0, a1)
            out.append((nxt, reward))
        return out
