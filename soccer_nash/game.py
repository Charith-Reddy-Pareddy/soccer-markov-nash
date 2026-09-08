"""Two-player soccer Markov game (CS 540 A10 geometry).

State: ``(x0, y0, x1, y1, b)`` where ``(x0, y0)`` is player 0's cell,
``(x1, y1)`` is player 1's cell and ``b in {0, 1}`` is the ball carrier.

Terminal states are encoded ``(-1, -1, -1, -1, winner)``.

Both players pick an action simultaneously. The move set is ``{U, D, L, R}`` by
default; ``n_actions=5`` adds ``STAND`` (stay put), which is Littman's (1994)
fifth action and the one his Figure 2 mixed equilibrium randomizes over. STAND
needs no special resolution code: a standing player simply targets its own cell,
and the existing "move into an occupied cell fails / transfers the ball" logic
does the rest.

Two resolution rules are supported:

* ``move_order="deterministic"`` -- the A10 rule: the carrier wins contested
  squares, swaps flip possession, a carrier blocked by a standing opponent
  loses the ball. Transitions are a pure function of the joint action.
* ``move_order="random"`` -- Littman's rule: the two moves are applied in a
  random order (each ordering with probability 1/2); a move into the other
  player's current cell fails and transfers the ball if the mover held it.
* ``move_order="coinflip"`` -- the A10 rule but a fair coin, not possession,
  decides who wins a contested square or a swap (each player with probability
  1/2). ``deterministic`` is this rule with the carrier always winning.

Two reward objectives (``scoring=``):

* ``"win"`` (default) -- the first goal ends the game, reward ``+1 / -1``. The
  value is ``P(player 0 wins) - P(player 1 wins)`` under optimal play; the game
  is a terminating zero-sum game and ``gamma < 1`` is only a solver device.
* ``"rate"`` -- a goal scores ``+1 / -1`` and **play continues** from a restart:
  the conceding team gets the ball at its own kickoff cell. The game runs to
  ``max_steps``. The value is the *expected discounted goal difference*, so
  ``gamma < 1`` is load-bearing, and conceding is no longer purely bad -- you
  get the ball back. See ``docs/reward.md``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import numpy as np

State = tuple[int, int, int, int, int]
Outcome = tuple[float, State, tuple[int, int]]

_DEFAULT_RNG = np.random.default_rng()


class Action(IntEnum):
    U = 0
    D = 1
    L = 2
    R = 3
    STAND = 4


# (dx, dy) per action. Up increases y.
_DELTA: dict[Action, tuple[int, int]] = {
    Action.U: (0, 1),
    Action.D: (0, -1),
    Action.L: (-1, 0),
    Action.R: (1, 0),
    Action.STAND: (0, 0),
}

#: the four movement actions -- the default move set
MOVE_ACTIONS: tuple[Action, ...] = (Action.U, Action.D, Action.L, Action.R)

# Joint-action order used by A10 Question 1: UU, UD, UL, UR, DU, ... (4x4 only).
JOINT_ACTIONS: list[tuple[Action, Action]] = [
    (a0, a1) for a0 in MOVE_ACTIONS for a1 in MOVE_ACTIONS
]


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


@dataclass(frozen=True)
class SoccerGame:
    width: int = 7
    height: int = 5
    goal_rows: tuple[int, ...] = (1, 2, 3)
    max_steps: int = 100
    move_order: str = "deterministic"
    # kickoff positions; ``None`` -> the centre row at opposite ends
    p0_start: tuple[int, int] | None = None
    p1_start: tuple[int, int] | None = None
    #: 4 -> {U, D, L, R}; 5 -> also STAND (Littman's fifth action)
    n_actions: int = 4
    #: "win" -> first goal ends the game; "rate" -> goal resets, play continues
    scoring: str = "win"

    def __post_init__(self) -> None:
        if self.move_order not in ("deterministic", "random", "coinflip"):
            raise ValueError(f"unknown move_order {self.move_order!r}")
        if self.n_actions not in (4, 5):
            raise ValueError(f"n_actions must be 4 or 5, got {self.n_actions}")
        if self.scoring not in ("win", "rate"):
            raise ValueError(f"scoring must be 'win' or 'rate', got {self.scoring!r}")
        for name, p in (("p0_start", self.p0_start), ("p1_start", self.p1_start)):
            if p is not None and not (
                0 <= p[0] < self.width and 0 <= p[1] < self.height
            ):
                raise ValueError(f"{name} {p} is off the {self.width}x{self.height} board")
        if (
            self.p0_start is not None
            and self.p1_start is not None
            and tuple(self.p0_start) == tuple(self.p1_start)
        ):
            raise ValueError("p0_start and p1_start must differ")

    # ------------------------------------------------------------------ basics
    def actions(self) -> list[Action]:
        """The move set: ``[U, D, L, R]``, plus ``STAND`` when ``n_actions == 5``."""
        return [Action(i) for i in range(self.n_actions)]

    def joint_actions(self) -> list[tuple[Action, Action]]:
        """Every ``(a0, a1)`` pair, row-major in the move set (A10 order for 4)."""
        acts = self.actions()
        return [(a0, a1) for a0 in acts for a1 in acts]

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
        """Kickoff: the ball is with player 0. Positions default to the centre
        row at opposite ends; set ``p0_start`` / ``p1_start`` to override."""
        mid = self.height // 2
        p0 = self.p0_start if self.p0_start is not None else (0, mid)
        p1 = self.p1_start if self.p1_start is not None else (self.width - 1, mid)
        return (p0[0], p0[1], p1[0], p1[1], 0)

    # ------------------------------------------------------------- transition
    def _target(
        self, player: int, pos: tuple[int, int], action: Action, has_ball: bool
    ) -> tuple[tuple[int, int], bool]:
        """Return (clamped target cell, scored?) for one player in isolation."""
        (x, y) = pos
        dx, dy = _DELTA[action]
        rx, ry = x + dx, y + dy

        # player 0 attacks the right edge (x = width), player 1 the left (x = -1)
        scored = has_ball and y in self.goal_rows and (
            (player == 0 and rx >= self.width) or (player == 1 and rx < 0)
        )

        return (_clamp(rx, 0, self.width - 1), _clamp(ry, 0, self.height - 1)), scored

    def _restart_state(self, conceding: int) -> State:
        """Kickoff after a goal: both players at their start cells, the ball with
        the team that just conceded."""
        mid = self.height // 2
        p0 = self.p0_start if self.p0_start is not None else (0, mid)
        p1 = self.p1_start if self.p1_start is not None else (self.width - 1, mid)
        return (p0[0], p0[1], p1[0], p1[1], conceding)

    def _score_result(self, winner: int) -> tuple[State, tuple[int, int], bool]:
        reward = (1, -1) if winner == 0 else (-1, 1)
        if self.scoring == "win":
            return (-1, -1, -1, -1, winner), reward, True
        # "rate": score the goal and play on from the restart
        return self._restart_state(1 - winner), reward, False

    def _resolve_with_winner(
        self,
        p0: tuple[int, int],
        p1: tuple[int, int],
        a0: Action,
        a1: Action,
        b: int,
        winner: int,
    ) -> tuple[State, tuple[int, int], bool]:
        """Resolve a joint action where ``winner`` wins any contest.

        The A10 deterministic rule is this with ``winner = b`` (the carrier).
        """
        (t0, s0) = self._target(0, p0, a0, has_ball=(b == 0))
        (t1, s1) = self._target(1, p1, a1, has_ball=(b == 1))

        # Only the carrier can score, so at most one of s0/s1 is true.
        if s0:
            return self._score_result(0)
        if s1:
            return self._score_result(1)

        pos = (p0, p1)
        tgt = (t0, t1)
        loser = 1 - winner

        if t0 == t1:
            # Contested cell: the winner moves in (unless the loser is standing
            # on it) and the loser takes the ball either way.
            n = [pos[0], pos[1]]
            if tgt[winner] != pos[loser]:
                n[winner] = tgt[winner]
            n0, n1 = n
            new_b = loser
        elif t0 == p1 and t1 == p0:
            # Swap: players exchange cells; the coin decides possession.
            n0, n1 = p1, p0
            new_b = loser
        else:
            n0, n1 = t0, t1
            new_b = b

        return (n0[0], n0[1], n1[0], n1[1], new_b), (0, 0), False

    def _resolve_deterministic(
        self,
        p0: tuple[int, int],
        p1: tuple[int, int],
        a0: Action,
        a1: Action,
        b: int,
    ) -> tuple[State, tuple[int, int], bool]:
        return self._resolve_with_winner(p0, p1, a0, a1, b, winner=b)

    def _resolve_sequential(
        self,
        p0: tuple[int, int],
        p1: tuple[int, int],
        a0: Action,
        a1: Action,
        b: int,
        first: int,
    ) -> tuple[State, tuple[int, int], bool]:
        pos = {0: p0, 1: p1}
        acts = {0: a0, 1: a1}
        ball = b

        for mover in (first, 1 - first):
            other = 1 - mover
            tgt, scored = self._target(
                mover, pos[mover], acts[mover], has_ball=(ball == mover)
            )
            if scored:
                return self._score_result(mover)
            if tgt == pos[other]:
                # Blocked by the other player's *current* cell (it may have
                # already moved this turn); a bump transfers the ball if the
                # mover was carrying it.
                if ball == mover:
                    ball = other
            else:
                pos[mover] = tgt

        (x0, y0), (x1, y1) = pos[0], pos[1]
        return (x0, y0, x1, y1, ball), (0, 0), False

    def transitions(self, state: State, a0: Action, a1: Action) -> list[Outcome]:
        """All ``(probability, next_state, (r0, r1))`` outcomes of a joint action."""
        if self.is_terminal(state):
            raise ValueError("transitions() called on a terminal state")

        x0, y0, x1, y1, b = state
        p0, p1 = (x0, y0), (x1, y1)

        if self.move_order == "deterministic":
            ns, reward, _ = self._resolve_deterministic(p0, p1, a0, a1, b)
            return [(1.0, ns, reward)]

        if self.move_order == "random":
            branches = [
                self._resolve_sequential(p0, p1, a0, a1, b, first)
                for first in (0, 1)
            ]
        else:  # coinflip
            branches = [
                self._resolve_with_winner(p0, p1, a0, a1, b, winner)
                for winner in (0, 1)
            ]

        merged: dict[tuple[State, tuple[int, int]], float] = {}
        for ns, reward, _ in branches:
            merged[(ns, reward)] = merged.get((ns, reward), 0.0) + 0.5
        return [(prob, ns, reward) for (ns, reward), prob in merged.items()]

    def step(
        self,
        state: State,
        a0: Action,
        a1: Action,
        rng: np.random.Generator | None = None,
    ) -> tuple[State, tuple[int, int], bool]:
        """Apply a joint action, sampling stochastic outcomes.

        Returns ``(next_state, (r0, r1), done)``.
        """
        outcomes = self.transitions(state, a0, a1)
        if len(outcomes) == 1:
            _, ns, reward = outcomes[0]
        else:
            rng = rng if rng is not None else _DEFAULT_RNG
            probs = [p for p, _, _ in outcomes]
            idx = int(rng.choice(len(outcomes), p=probs))
            _, ns, reward = outcomes[idx]
        return ns, reward, self.is_terminal(ns)

    def successors(self, state: State) -> list[list[Outcome]]:
        """Outcome list for every joint action, in A10 order."""
        return [self.transitions(state, a0, a1) for a0, a1 in JOINT_ACTIONS]


@dataclass(frozen=True)
class A10SoccerGame(SoccerGame):
    """The exact CS 540 A10 environment: deterministic move resolution only.

    This is the game the "pure saddle at every state" result is measured on.
    The interpreted collision sub-cases are listed in ``docs/assumptions.md``.
    Research variants (`move_order` "random" / "coinflip") are *different game
    definitions* -- use ``SoccerGame`` for those, not this class.

    On the standard 7x5 board the kickoff is the one the A10 page assigns this
    student's netID (player 0 at (0, 1), player 1 at (6, 3)); a different netID
    gets different positions, so pass ``p0_start`` / ``p1_start`` for yours. The
    result "every stage game has a pure saddle" is over *all* states and does not
    depend on the kickoff, but ``V(kickoff)`` and the Part 2 Q8 trajectory do.
    """

    #: this student's ID-specific kickoff on the 7x5 board (A10 page)
    A10_KICKOFF: ClassVar[tuple[tuple[int, int], tuple[int, int]]] = ((0, 1), (6, 3))

    move_order: str = "deterministic"

    def __post_init__(self) -> None:
        if self.move_order != "deterministic" or self.scoring != "win":
            raise ValueError(
                "A10SoccerGame is the exact assignment environment; use "
                "SoccerGame(move_order=..., scoring=...) for research variants"
            )
        super().__post_init__()

    def initial_state(self) -> State:
        if (
            (self.width, self.height) == (7, 5)
            and self.p0_start is None
            and self.p1_start is None
        ):
            (x0, y0), (x1, y1) = self.A10_KICKOFF
            return (x0, y0, x1, y1, 0)
        return super().initial_state()
